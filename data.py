from torch.utils.data import Dataset, DataLoader


def get_dataloaders():
    dataset = PromptCompDataset()
    return DataLoader(dataset)


class PromptCompDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return super().__getitem__(index)

    def collate_fn(self, batch):
        return batch