import torch
from data_preprocessing import return_dataloaders
from computer_vision.auteencoders.models.AutoEncoder import AutoEncoder_mnist
train_dataloader, val_dataloader, test_dataloader = return_dataloaders(128, 128, 3)

print(next(iter(train_dataloader)))
model = AutoEncoder_mnist()
print(model)
