import numpy as np
import torch

def get_Tossingdata():
    data = np.loadtxt('/home/elhamod/projects/PID-GAN/PID-GAN/datasets/tossing_trajectories.txt')
    x = data[:, 0:6]
    labels = data[:, 6:]
    tr_frac = 0.4

    # training and test split
    n_obs = int(tr_frac * x.shape[0])
    train_x , train_y = x[:n_obs,:] , labels[:n_obs, :] 
    test_x , test_y = x[n_obs:,:] , labels[n_obs:, :]


    return train_x, test_x, train_y, test_y

class Tossingdata_bank():
    def __init__(self):
        self.data = get_Tossingdata()
    
    def get_data(self, with_normalization=False):
        train_x, test_x, train_y, test_y = self.data
        Xmean = None
        Xstd = None
        Ymean = None
        Ystd = None

        if with_normalization:
            Xmean, Xstd = train_x.mean(0), train_x.std(0)
            Ymean, Ystd = train_y.mean(0), train_y.std(0)

            
            train_x = (train_x - Xmean) / Xstd
            test_x = (test_x - Xmean) / Xstd
            train_y = (train_y- Ymean) / Ystd
            test_y = (test_y - Ymean) / Ystd

        return train_x, test_x, train_y, test_y, Xmean, Xstd, Ymean, Ystd

databank = Tossingdata_bank()

    
def PIDPINN_criterion(args, loss_key='train_loss'):
    def physics_loss(inp, out, stat_inp = [0,1], stat_out = [0,1], t = 0.0333333):  #stat [0] = mean, stat [1] = std
    
        fps = 30
        stat_inp = torch.Tensor(stat_inp).to(inp)
        stat_out = torch.Tensor(stat_out).to(inp)
        inp = inp * stat_inp[1] + stat_inp[0]
        out = out * stat_out[1] + stat_out[0]


        v_x = (inp[:,1] - inp[:, 0])/t
        v_y = (inp[:,4] - inp[:, 3] + 0.5 * 9.8 * t*t)/t

        pred_x = out[:, 0:15]
        pred_y = out[:, 15:]

        g = 9.8

        t = np.arange(0, 1/fps * (inp.shape[1] + out.shape[1]) * 0.5, 1/fps)

        t = np.repeat([t[3:]], inp.shape[0], axis=0)
        t_s = torch.tensor(t).float().to(inp)

        x_0 = inp[:, 0].repeat(15, 1).T
        y_0 = inp[:, 3].repeat(15, 1).T
        v_x_t = v_x.repeat(15, 1).T
        v_y_t = v_y.repeat(15, 1).T

        x_loc_pred = x_0 + v_x_t * t_s
        y_loc_pred = y_0 + v_y_t * t_s - 0.5 * g * t_s**2


        loss = 0.5*(((pred_x  - x_loc_pred)**2).mean(dim=1) + ((pred_y  - y_loc_pred)**2).mean(dim=1))
        return loss

    def losses_fn(y, net_x, x_f, net_x_f, Xmean=None, Xstd=None, Ymean=None, Ystd=None):
        if loss_key == 'train_loss':
            def phy_loss(x_f, net_x_f, Xmean, Xstd, Ymean, Ystd):
                return torch.mean(torch.abs(physics_loss(x_f, net_x_f, [Xmean, Xstd], [Ymean, Ystd])))
            
            print('a')
            
            losses = [torch.nn.functional.mse_loss(net_x, y).item(), phy_loss(x_f, net_x_f, Xmean, Xstd, Ymean, Ystd).item()]
            if args.whichloss is None:
                return losses
            
            print('b')

            return [losses[args.task_name.index(args.whichloss)]]
        else:
            return [torch.nn.functional.mse_loss(net_x, y).item()]
  
        # mse = torch.nn.functional.mse_loss(self.net.forward(self.train_x), self.train_y).item()
        # phy =  torch.mean(torch.abs(self.physics_loss(self.x_f, self.net.forward(self.x_f),  [self.Xmean, self.Xstd], [self.Ymean, self.Ystd]))).item()
        # loss_weights = [1.0]*self.task_num if 'weights' not in self.kwargs.keys() else self.kwargs['weights']
        # loss = torch.sum(torch.mul(torch.tensor([mse, phy]), torch.tensor(loss_weights))).item()
        
    return losses_fn