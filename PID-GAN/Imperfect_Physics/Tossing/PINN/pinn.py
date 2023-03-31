import sys
sys.path.insert(0, '/home/elhamod/projects/PID-GAN/PID-GAN/Imperfect_Physics/')
from earlystopping import EarlyStopping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import pandas as pd

class Tossing_PINN():
    def __init__(self, train_x, train_y, test_x, test_y, net, device, nepochs, lambda_val=None):
        super(Tossing_PINN, self).__init__()
        
        # Normalize data
        self.Xmean, self.Xstd = train_x.mean(0), train_x.std(0)
        self.Ymean, self.Ystd = train_y.mean(0), train_y.std(0)
        
        self.train_x = (train_x - self.Xmean) / self.Xstd
        self.test_x = (test_x - self.Xmean) / self.Xstd
        self.train_y = (train_y- self.Ymean) / self.Ystd
        self.test_y = (test_y - self.Ymean) / self.Ystd
        
        self.net = net
        
        self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas = (0.5, 0.999))
        
        self.device = device
        
        # numpy to tensor
        self.train_x = torch.tensor(self.train_x, requires_grad=True).float().to(self.device)
        self.train_y = torch.tensor(self.train_y, requires_grad=True).float().to(self.device)
        self.test_x = torch.tensor(self.test_x, requires_grad=True).float().to(self.device)
        self.test_y = torch.tensor(self.test_y, requires_grad=True).float().to(self.device)
        self.x_f = torch.cat([self.train_x, self.test_x], dim = 0)
        
        self.nepochs = nepochs
        self.lambda_val = lambda_val

        self.model_save_prefix = None
        self.every_epoch = 100
        
        self.batch_size = 64
        num_workers = 4
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x,self.train_y)), batch_size=self.batch_size, shuffle=shuffle, generator=torch.Generator(device=self.device)
        )
    
    def set_model_save(self, model_save_prefix):
        self.model_save_prefix = model_save_prefix

    def uncertainity_estimate(self, x, num_samples, stat = [0,1]):
        outputs = np.stack([self.net.forward(x).cpu().detach().numpy()*stat[1]+stat[0] for i in range(num_samples)], axis = 0)
        y_mean = outputs.mean(axis=0)
        y_variance = outputs.var(axis=0)
        y_std = np.sqrt(y_variance)
        return y_mean, y_std
    
    def physics_loss(self, inp, out, stat_inp = [0,1], stat_out = [0,1], t = 0.0333333):  #stat [0] = mean, stat [1] = std
    
        fps = 30
        stat_inp = torch.Tensor(stat_inp).to(self.device)
        stat_out = torch.Tensor(stat_out).to(self.device)
        inp = inp * stat_inp[1] + stat_inp[0]
        out = out * stat_out[1] + stat_out[0]


        v_x = (inp[:,1] - inp[:, 0])/t
        v_y = (inp[:,4] - inp[:, 3] + 0.5 * 9.8 * t*t)/t

        pred_x = out[:, 0:15]
        pred_y = out[:, 15:]

        g = 9.8

        t = np.arange(0, 1/fps * (inp.shape[1] + out.shape[1]) * 0.5, 1/fps)

        t = np.repeat([t[3:]], inp.shape[0], axis=0)
        t_s = torch.tensor(t).float().to(self.device)

        x_0 = inp[:, 0].repeat(15, 1).T
        y_0 = inp[:, 3].repeat(15, 1).T
        v_x_t = v_x.repeat(15, 1).T
        v_y_t = v_y.repeat(15, 1).T

        x_loc_pred = x_0 + v_x_t * t_s
        y_loc_pred = y_0 + v_y_t * t_s - 0.5 * g * t_s**2


        loss = 0.5*(((pred_x  - x_loc_pred)**2).mean(dim=1) + ((pred_y  - y_loc_pred)**2).mean(dim=1))
        return loss
    
    
    
    def plot_lines(self, df, x_column, y_columns, title, log=False):
        plt.figure()
        # Create a line plot
        for y_column in y_columns:
            plt.plot(df[x_column], df[y_column], label=y_column) # , marker='o'

        plt.xlabel(x_column)
        plt.ylabel("Losses")
        plt.title(title)
        if log:
            plt.yscale("log")
        plt.legend()

        save_path = f"{self.model_save_prefix}-{title}.pdf"
        plt.savefig(save_path)
    
    def save_model(self, epoch, last=False):
        save_path = f"{self.model_save_prefix}-{epoch if not last else 'last'}"+".pt"
        torch.save(self.net.state_dict(), save_path)

        if last:
            def unpack_nested_dict(nested_dict):
                unpacked_dict = {}
                for outer_key, value in nested_dict.items():
                    if isinstance(value, dict):
                        for inner_key, inner_value in value.items():
                            unpacked_dict[f'{outer_key}_{inner_key}'] = inner_value
                    else:
                        unpacked_dict[outer_key] = value
                return unpacked_dict
            loss_df = [pd.DataFrame.from_dict(unpack_nested_dict(d), orient='index').T for d in self.loss_records]
            loss_df = pd.concat(loss_df, ignore_index=True)

            save_path = f"{self.model_save_prefix}.csv"
            loss_df.to_csv(save_path, index=False)

            self.plot_lines(loss_df, "epoch", ['train_loss_mse','train_loss_phy','train_loss_total'], 'training losses', log=True)
            self.plot_lines(loss_df, "epoch", ['test_loss_mse'], 'test loss', log=True)
            self.plot_lines(loss_df, "epoch", ['train_loss_mse','test_loss_mse'], 'training vs test losses', log=True)
            self.plot_lines(loss_df, "epoch", ['weight_mse','weight_phy'], 'weights')



    
    def train(self):
        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)
        TOT_loss = np.zeros(self.nepochs)

        self.loss_records = []

        mse = torch.nn.functional.mse_loss(self.net.forward(self.train_x), self.train_y).item()
        net_x_f = self.net.forward(self.x_f)
        phy =  torch.mean(torch.abs(self.physics_loss(self.x_f, net_x_f,  [self.Xmean, self.Xstd], [self.Ymean, self.Ystd]))).item()
        loss_weights = [1.0]*self.task_num if 'weights' not in self.kwargs.keys() else self.kwargs['weights']
        loss = torch.sum(torch.mul(torch.tensor([mse, phy]), torch.tensor(loss_weights))).item()
        
        dict = {
                'epoch':  0,
                'train_loss': {
                    'mse': mse,
                    'phy': phy, #NOTE: test_loss does not have phy because we use all data for it in train_loss
                    'total': loss,
                },
                'test_loss': {
                    'mse': torch.nn.functional.mse_loss(self.net.forward(self.test_x), self.test_y).item()
                },
                'weight': {
                    'mse': loss_weights[0],
                    'phy': loss_weights[1],
                }
            }
        self.loss_records.append(dict)

        earlystopping = EarlyStopping(self.net, patience=2000)
        earlystopping.on_train_begin()
        for epoch in range(self.nepochs):
            self.epoch = epoch

            if earlystopping.stop_training:
                break

            for i, (x, y) in enumerate(self.train_loader):

                self.net_optimizer.zero_grad()
                y_pred = self.net.forward(x)

                y_f = self.net.forward(self.x_f)

                phy_loss = torch.mean(torch.abs(self.physics_loss(self.x_f, y_f,  [self.Xmean, self.Xstd], [self.Ymean, self.Ystd])))
                mse_loss = torch.nn.functional.mse_loss(y_pred, y)

                losses = torch.stack((mse_loss, phy_loss))
                loss_weights = self.backward(losses, **self.kwargs['weight_args'])
                loss = torch.sum(torch.mul(losses, torch.tensor(loss_weights)))

                self.net_optimizer.step()

                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                PHY_loss[epoch] += phy_loss.detach().cpu().numpy()
                TOT_loss[epoch] += loss.detach().cpu().numpy()

            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)
            TOT_loss[epoch] = TOT_loss[epoch] / len(self.train_loader)

            earlystopping.on_epoch_end(epoch, TOT_loss[epoch])
            
            self.train_loss_buffer[:, epoch] = [MSE_loss[epoch], PHY_loss[epoch]]


            if (epoch % self.every_epoch == 0):
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch] ))

                    # save model
                self.save_model(epoch)

            dict = {
                'epoch':  epoch + 1,
                'train_loss': {
                    'total': TOT_loss[epoch],
                    'phy': PHY_loss[epoch],
                    'mse': MSE_loss[epoch],
                },
                'test_loss': {
                    'mse': torch.nn.functional.mse_loss(self.net.forward(self.test_x), self.test_y).item()
                },
                'weight': {
                    'mse': loss_weights[0],
                    'phy': loss_weights[1],
                }
            }
                
            self.loss_records.append(dict)

        self.save_model(epoch, last=True)

        earlystopping.on_train_end()