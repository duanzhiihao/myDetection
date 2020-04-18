import torch.optim


def get_optimizer(name, params, lr, cfg):
    if name == 'SGDMR':
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer
