import torch.optim


def get_optimizer(name, params, lr, cfg):
    if name == 'SGDMR':
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        raise NotImplementedError
    return optimizer