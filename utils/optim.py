import numpy as np
import torch.optim


def get_optimizer(name, params, lr, global_cfg):
    if name == 'SGDM':
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    elif name == 'SGDMN':
        optimizer = torch.optim.SGD(params, lr, momentum=0.937, nesterov=True)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        raise NotImplementedError
    return optimizer


def lr_warmup(i, warm_up=1000):
    '''
    Step schedular
    '''
    if i < warm_up:
        factor = i / warm_up
    else:
        factor = 1
    # elif i < 70000:
    #     factor = 0.5
    # elif i < 90000:
    #     factor = 0.25
    # elif i < 100000:
    #     factor = 0.1
    # elif i < 200000:
    #     factor = 1
    # else:
    #     factor = 0.01
    return factor


def cosine(i, epochs=300):
    '''
    Cosine schedular
    '''
    factor = (((1 + np.cos(i * np.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1
    return factor
