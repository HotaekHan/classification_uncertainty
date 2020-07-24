import numpy as np
import torch


class scheduled_optim(object):
    def __init__(self, params, config):
        self.optim_params = params
        self.config = config
        self.base_lr = float(config['optimizer']['lr'])
        self.max_epoch = float(config['params']['epoch'])
        self.warmup_epoch = float(self.config['optimizer']['warmup_epoch'])
        self.warmup_fatcor = float(self.config['optimizer']['warmup_factor'])
        self.exp_gamma = float(self.config['optimizer']['exp_gamma'])

    def construct_optimizer(self):
        """Constructs the optimizer.

        Note that the momentum update in PyTorch differs from the one in Caffe2.
        In particular,

            Caffe2:
                V := mu * V + lr * g
                p := p - V

            PyTorch:
                V := mu * V + g
                p := p - lr * V

        where V is the velocity, mu is the momentum factor, lr is the learning rate,
        g is the gradient and p are the parameters.

        Since V is defined independently of the learning rate in PyTorch,
        when the learning rate is changed there is no need to perform the
        momentum correction by scaling V (unlike in the Caffe2 case).
        """
        # if cfg.BN.USE_CUSTOM_WEIGHT_DECAY:
        #     # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        #     p_bn = [p for n, p in model.named_parameters() if "bn" in n]
        #     p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
        #     optim_params = [
        #         {"params": p_bn, "weight_decay": cfg.BN.CUSTOM_WEIGHT_DECAY},
        #         {"params": p_non_bn, "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        #     ]
        # else:
        #     optim_params = model.parameters()

        if self.config['optimizer']['use_adam'] is True:
            optim = torch.optim.Adam(params=self.optim_params, lr=self.base_lr)
        else:
            optim = torch.optim.SGD(
                params=self.optim_params,
                lr=self.base_lr,
                momentum=float(self.config['optimizer']['momentum']),
                weight_decay=float(self.config['optimizer']['weight_decay']),
                dampening=float(self.config['optimizer']['dampening']),
                nesterov=self.config['optimizer']['nesterov'],
            )
        return optim


    # def lr_fun_steps(self, cur_epoch):
    #     """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    #     ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    #     return self.base_lr * (cfg.OPTIM.LR_MULT ** ind)

    def lr_fun_exp(self, cur_epoch):
        """Exponential schedule (cfg.OPTIM.LR_POLICY = 'exp')."""
        return self.base_lr * (self.exp_gamma ** cur_epoch)


    def lr_fun_cos(self, cur_epoch):
        """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
        base_lr, max_epoch = self.base_lr, self.max_epoch
        return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / max_epoch))


    def get_epoch_lr(self, cur_epoch):
        """Retrieves the lr for the given epoch according to the policy."""
        if self.config['optimizer']['lr_policy'] == 'cos':
            lr = self.lr_fun_cos(cur_epoch)
        elif self.config['optimizer']['lr_policy'] == 'exp':
            lr = self.lr_fun_exp(cur_epoch)
        else:
            raise NotImplementedError('Not implemented lr_policy : ' + str(self.config['optimizer']['lr_policy']))

        # Linear warmup
        if cur_epoch < self.warmup_epoch:
            alpha = cur_epoch / self.warmup_epoch
            warmup_factor = self.warmup_fatcor * (1.0 - alpha) + alpha
            lr *= warmup_factor
        return lr


    def set_lr(self, optimizer, new_lr):
        """Sets the optimizer lr to the specified value."""
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


if __name__ == '__main__':
    import torch.nn as nn
    import matplotlib.pyplot as plt

    net = nn.Linear(10, 100)
    config = {'optimizer':
                  {'lr':0.4,
                   'momentum':0.9,
                   'weight_decay':1e-4,
                   'dampening':0.0,
                   'nesterov':False,
                   'warmup_factor':0.1,
                   'warmup_epoch':5,
                   'exp_gamma':0.1,
                   'lr_policy':'cos',
                   'use_adam':False},
              'params':
                  {'epoch':300}}
    optim = scheduled_optim(params=net.parameters(), config=config)
    optimizer = optim.construct_optimizer()

    list_lr = list()
    for iter_epoch in range(config['params']['epoch']):
        lr = optim.get_epoch_lr(iter_epoch)
        optim.set_lr(optimizer, lr)
        list_lr.append(lr)

    plt.plot(list_lr)
    plt.show()