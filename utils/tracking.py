import numpy as np
import torch


def linear_predict(history):
    '''
    Args:
        history: np.array or torch.tensor
    '''
    assert isinstance(history, (np.ndarray, torch.Tensor))
    assert len(history.shape) == 2 and history.shape[1] >= 4
    if history.shape[0] == 1:
        return history[0, :]
    t_1 = history[-1,:]
    t_2 = history[-2,:]
    v = t_1 - t_2
    pred = t_1 + v
    assert len(pred.shape) == 1
    return pred