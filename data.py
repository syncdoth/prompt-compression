from torch.utils.data import Dataset, DataLoader


def get_dataloaders(
    batch_size=1,
    eval_batch_size=1,
    **dataset_kwargs,
):
    dataloaders = {}
    for mode in ('train', 'valid', 'test'):
        dataset = PromptCompDataset(**dataset_kwargs, mode=mode)
        loader = DataLoader(dataset,
                            batch_size=batch_size if mode == 'train' else eval_batch_size,
                            collate_fn=dataset.make_batch)
        dataloaders[mode] = loader

    return dataloaders


class PromptCompDataset(Dataset):

    def __init__(
        self,
        mode='train',
        **kwargs,
    ):
        self.mode = mode
        for kw, arg in kwargs:
            self.__setattr__(kw, arg)
        # TODO: add special actions

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def make_batch(self, batch):
        """collate_fn to be passed to the torch.utils.data.DataLoader"""
        raise NotImplementedError
