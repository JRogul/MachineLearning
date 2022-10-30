import torch
from data_preprocessing import return_dataloaders

train_dataloader, val_dataloader, test_dataloader = return_dataloaders(128, 128, 3)

print(next(iter(train_dataloader)))
