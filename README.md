# PID-GAN

This repository contains the official implementation for the paper [PID-GAN: A GAN Framework based on a Physics-informed Discriminator for Uncertainty Quantification with Physics](https://arxiv.org/pdf/2106.02993.pdf). 

### Abstract
As applications of deep learning (DL) continue to seep into critical scientific use-cases, the importance of performing uncertainty quantification (UQ) with DL has become more pressing than ever before. In scientific applications, it is also important to inform the learning of DL models with knowledge of physics of the problem to produce physically consistent and generalized solutions. This is referred to as the emerging field of physics-informed deep learning (PIDL). We consider the problem of developing PIDL formulations that can also perform UQ. To this end, we propose a novel physics-informed GAN architecture, termed PID-GAN, where the knowledge of physics is used to inform the learning of both the generator and discriminator models, making ample use of unlabeled data instances. We show that our proposed PID-GAN framework does not suffer from imbalance of generator gradients from multiple loss terms as compared to state-of-the-art. We also empirically demonstrate the efficacy of our proposed framework on a variety of case studies involving benchmark physics-based PDEs as well as imperfect physics.
