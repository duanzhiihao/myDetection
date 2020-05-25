raise NotImplementedError()

from torch.utils.data import Dataset, DataLoader


class InfiniteDataset(Dataset):
    '''
    Base dataset class that has a infinite length
    '''
    def __init__(self):
        """Should be overridden by all subclasses."""
        raise NotImplementedError()

    def __len__(self):
        '''Dummy function'''
        return 1000000000

    def __getitem__(self, _):
        """Should be overridden by all subclasses."""
        raise NotImplementedError()

    def to_iter(self, **kwargs):
        # self.dataloader = DataLoader(self, **kwargs)
        self.iter = iter(DataLoader(self, **kwargs))
    
    def get_next(self):
        data = next(self.iter)
        return data
