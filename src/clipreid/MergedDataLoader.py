import random
from torch.utils.data import DataLoader


class MergedDataLoader(DataLoader):
    def __init__(self, loader1, loader2, *args, **kwargs):
        """ Initialize with two dataloaders and pass additional args to DataLoader """
        assert loader1.batch_size == loader2.batch_size, "Batch sizes must be the same!"
        self.loader1 = loader1
        self.loader2 = loader2
        super().__init__([], *args, **kwargs)  # Initialize base DataLoader

    def __iter__(self):
        """ Reset dataloaders at the start of each epoch """
        self.iter1 = iter(self.loader1)
        self.iter2 = iter(self.loader2)
        self.loaders = [self.iter1, self.iter2]
        # self.loaders = [self.iter1]
        return self

    def __next__(self):
        """ Randomly choose between the two dataloaders and fetch the next batch """
        if not self.loaders:
            raise StopIteration  # Stop when both loaders are exhausted

        chosen_loader = random.choice(self.loaders)
        try:
            return next(chosen_loader)
        except StopIteration:
            self.loaders.remove(chosen_loader)  # Remove exhausted loader
            if self.loaders:  # If any loaders remain, try again
                return self.__next__()
            else:
                raise StopIteration  # Let Python naturally handle StopIteration

    def __len__(self):
        """ Return the sum of the lengths of both dataloaders """
        return len(self.loader1) + len(self.loader2)

# # Example Usage
# dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)
# dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)

# merged_loader = MergedDataLoader(dataloader1, dataloader2)

# for epoch in range(num_epochs):
#     for batch in merged_loader:
#         # Training step
#         pass
