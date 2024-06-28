import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = torch.utils.data.Subset(dataset, indices=train_indices)
    val_dataset = torch.utils.data.Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset


def __get_normalizer(dataset_name):
    if dataset_name == 'imagenet':
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset_name == 'cifar10':
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset_name == 'cifar100':
        return transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    elif dataset_name == 'svhn':
        return transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052))
    else:
        raise ValueError('load_data does not support dataset %s' % dataset_name)


def __get_dataset(dataset_name, dataset_path):
    normalize = __get_normalizer(dataset_name)
    
    if dataset_name == 'imagenet':
        pass
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if dataset_name == 'cifar10':
            train_set = datasets.CIFAR10(dataset_path, train=True, transform=train_transform, download=True)
            test_set = datasets.CIFAR10(dataset_path, train=False, transform=val_transform, download=True)
        elif dataset_name == 'cifar100':
            train_set = datasets.CIFAR100(dataset_path, train=True, transform=train_transform, download=True)
            test_set = datasets.CIFAR100(dataset_path, train=False, transform=val_transform, download=True)
        elif dataset_name == 'svhn':
            train_set = datasets.SVHN(dataset_path, split='train', transform=train_transform, download=True)
            test_set = datasets.SVHN(dataset_path, split='test', transform=val_transform, download=True)
    else:
        raise ValueError('load_data does not support dataset %s' % dataset_name)
    
    return train_set, test_set


def load_data(dataset, batch):
    train_set, test_set = __get_dataset(dataset, './data')
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch, num_workers=4, pin_memory=True)

    return train_loader, test_loader
