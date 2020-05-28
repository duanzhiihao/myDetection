import torch
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

    @staticmethod
    def collate_func(batch):
        raise NotImplementedError()

    def to_iterator(self, **kwargs):
        self.iterator = iter(DataLoader(self, collate_fn=self.collate_func,
                                        **kwargs))
        self._iter_args = kwargs

    def get_next(self):
        assert hasattr(self, 'to_iterator'), 'Please call to_iterator() first'
        try:
            data = next(self.iterator)
        except StopIteration:
            print(f'Warning: loaded {len(self)} images!')
            self.to_iterator(**self._iter_args)
            data = next(self.iterator)
        return data