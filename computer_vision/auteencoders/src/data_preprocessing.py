import torch
from torchvision import transforms
import torchvision.datasets as datasets


def return_dataloaders(train_batch_size, val_batch_size, test_batch_size):

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(root="C:\\Users\\giga\\Desktop\\gitt\\MachineLearning\\computer_vision\\auteencoders\\data",
                                   train=True, download=False, transform=train_transforms)
    val_dataset = datasets.MNIST(root='C:\\Users\\giga\\Desktop\\gitt\\MachineLearning\\computer_vision\\auteencoders\\data',
                                 train=False, download=False, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch_size,
                                                   num_workers=0,
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=val_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=test_batch_size,
                                                  shuffle=False,
                                                  num_workers=0)
    return train_dataloader, val_dataloader, test_dataloader

