'''
This code is based on https://github.com/bboylyg/NAD

The original license:
License CC BY-NC

The update include:
    1. decompose the function structure and add more normalization options
    2. add more dataset options, and compose them into dataset_and_transform_generate

# idea : use args to choose which dataset and corresponding transform you want
'''
import logging
import os
import random
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from typing import Any, Callable, List, Optional, Union, Tuple
from PIL import ImageFilter, Image



def get_num_classes(dataset_name: str) -> int:
    # idea : given name, return the number of class in the dataset
    if dataset_name in ["mnist", "fmnist", "stl10", "cifar10"]:
        num_classes = 10
    elif dataset_name == "utkface":
        num_classes = 4
    elif dataset_name == "gtsrb":
        num_classes = 43
    elif dataset_name == "celeba":
        num_classes = 8
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'tiny':
        num_classes = 200
    elif dataset_name == 'imagenet':
        num_classes = 1000
    else:
        raise Exception("Invalid Dataset")
    return num_classes


def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    # idea : given name, return the image size of images in the dataset
    if dataset_name == "fmnist":
        # input_height = 28
        # input_width = 28
        input_height = 64
        input_width = 64
        input_channel = 1
    elif dataset_name == "utkface":
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == "stl10":
        # input_height = 96
        # input_width = 96
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == "cifar10":
        # input_height = 32
        # input_width = 32
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == "gtsrb":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "mnist":
        input_height = 28
        input_width = 28
        input_channel = 1
    elif dataset_name == "celeba":
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == 'cifar100':
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == 'tiny':
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == 'imagenet':
        input_height = 224
        input_width = 224
        input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    return input_height, input_width, input_channel


def get_dataset_normalization(dataset_name):
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # idea : given name, return the default normalization of images in the dataset
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = (transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        dataset_normalization = (transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]))
    elif dataset_name == "mnist":
        dataset_normalization = (transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'imagenet':
        dataset_normalization = (
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    # else:
    #     raise Exception("Invalid Dataset")
    return dataset_normalization


def get_dataset_denormalization(normalization: transforms.Normalize):
    print("*******************normalization************************", normalization)
    mean, std = normalization.mean, normalization.std
    print("*******************mean************************", mean, std)
    print("*******************mean.__len__()************************", mean.__len__())

    if mean.__len__() == 1:
        mean = (- mean[0],)
    else:  # len > 1
        mean = [-i for i in mean]

    if std.__len__() == 1:
        std = (1 / std[0],)
    else:  # len > 1
        std = [1 / i for i in std]

    # copy from answer in
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    # user: https://discuss.pytorch.org/u/svd3

    if mean.__len__() == 1:
        invTrans = transforms.Compose([
            transforms.Normalize(mean=[0.],
                                std=std),
            transforms.Normalize(mean=mean,
                                std=[1.]),
        ])
    else:
        invTrans = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                std=std),
            transforms.Normalize(mean=mean,
                                std=[1., 1., 1.]),
        ])

    return invTrans


def get_transform_prefetch(dataset_name, input_height, input_width, train=True, prefetch=False):
    # idea : given name, return the final implememnt transforms for the dataset
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=4))
        # transforms_list.append(transforms.RandomRotation(10))
        if dataset_name == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())
    if not prefetch:
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms.Compose(transforms_list)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR.

    Borrowed from https://github.com/facebookresearch/moco/blob/master/moco/loader.py.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x


def get_transform_self(dataset_name, input_height, input_width, train=True, prefetch=False):
    # idea : given name, return the final implememnt transforms for the dataset during self-supervised learning
    transforms_list = []
    transforms_list.append(transforms.Resize((input_height, input_width)))
    if train:

        transforms_list.append(
            transforms.RandomResizedCrop(size=(input_height, input_width), scale=(0.2, 1.0), ratio=(0.75, 1.3333),
                                         interpolation=3))
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transforms_list.append(transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=[0.6, 1.4],
                                                                                                  contrast=[0.6, 1.4],
                                                                                                  saturation=[0.6, 1.4],
                                                                                                  hue=[-0.1, 0.1])]),
                                                      p=0.8))
        transforms_list.append(transforms.RandomGrayscale(p=0.2))
        transforms_list.append(transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5))

    if not prefetch:
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms.Compose(transforms_list)


def get_transform(dataset_name, train=False):
    # idea : given name, return the final implememnt transforms for the dataset
    transforms_list = []
    if dataset_name.lower() == "utkface":
        # transforms_list.append(transforms.Resize((64, 64)))
        if train:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transforms_list)

    elif dataset_name.lower() == "stl10":
        # transforms_list.append(transforms.Resize((64, 64)))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform = transforms.Compose(transforms_list)

    elif dataset_name.lower() == "cifar10":
        # transforms_list.append(transforms.Resize((64, 64)))
        if train:
            # transforms.RandomCrop((64, 64), padding=4)
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transforms_list)

    else:    #   dataset_name.lower() == "fmnist"
        
        # transforms_list.append(transforms.Resize((64, 64)))
        if train:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
        transform = transforms.Compose(transforms_list)

    
    # transforms_list = []
    # transforms_list.append(transforms.Resize((input_height, input_width)))
    # if train:
    #     transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=random_crop_padding))
    #     # transforms_list.append(transforms.RandomRotation(10))
    #     if dataset_name == "cifar10":
    #         transforms_list.append(transforms.RandomHorizontalFlip())

    # transforms_list.append(transforms.ToTensor())
    # transforms_list.append(get_dataset_normalization(dataset_name))
    # return transforms.Compose(transforms_list)
    return transform

class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.processed_path = os.path.join(self.root, 'UTKFace/processed/')
        self.files = os.listdir(self.processed_path)
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            # txt_file_path = os.path.join(self.processed_path, txt_file)
            # print(txt_file)
            image_name = txt_file.split('jpg')[0]
            attrs = image_name.split('_')
            if len(attrs) < 4 or int(attrs[2]) >= 4 or '' in attrs:
                continue
            self.lines.append(image_name+'jpg')
        # print(self.lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(
            self.root, 'UTKFace/raw/', self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)

            else:
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target


def dataset_and_transform_generate(args):
    '''
    # idea : given args, return selected dataset, transforms for both train and test part of data.
    :param args:
    :return: clean dataset in both train and test phase, and corresponding transforms

    1. set the img transformation
    2. set the label transform
    '''
    train_img_transform = get_transform(args.dataset, train=True)
    test_img_transform = get_transform(args.dataset, train=False)
    # print("************dataset_and_transform_generate***************")
    # print(args.dataset)
    # if not args.dataset.startswith('test'):
    #     train_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=True)
    #     test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    # else:
    #     # test folder datset, use the mnist transform for convenience
    #     train_img_transform = get_transform('mnist', *(args.img_size[:2]), train=True)
    #     test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    train_label_transform = None
    test_label_transform = None

    train_dataset_without_transform, test_dataset_without_transform = None, None

    if (train_dataset_without_transform is None) or (test_dataset_without_transform is None):

        if args.dataset == 'mnist':
            from torchvision.datasets import MNIST
            train_dataset_without_transform = MNIST(
                args.dataset_path,
                train=True,
                transform=None,
                download=False)
            test_dataset_without_transform = MNIST(
                args.dataset_path,
                train=False,
                transform=None,
                download=False)
            
        elif args.dataset.lower() == "utkface":
            # transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((64, 64)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # ])
            dataset = UTKFaceDataset(root=args.dataset_path, attr='race',transform=transforms.Resize((64, 64)))

        elif args.dataset.lower() == "fmnist":
            # transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            # transforms.ToTensor(),
            # # transforms.Normalize((0.1307), (0.3081))
            # transforms.Normalize((0.5), (0.5))
            # ])
            train_set = torchvision.datasets.FashionMNIST(
                root=args.dataset_path, train=True, download=False, transform=transforms.Resize((64, 64)))
            test_set = torchvision.datasets.FashionMNIST(
                root=args.dataset_path, train=False, download=False, transform=transforms.Resize((64, 64)))
            dataset = train_set + test_set

        elif args.dataset.lower() == "stl10":
            # transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                      (0.2023, 0.1994, 0.2010)),
            # ])
            train_set = torchvision.datasets.STL10(
                root=args.dataset_path, split='train', transform=transforms.Resize((64, 64)))
            test_set = torchvision.datasets.STL10(
                root=args.dataset_path, split='test', transform=transforms.Resize((64, 64)))
            dataset = train_set + test_set

        elif args.dataset.lower() == 'cifar10':
            # transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((64, 64)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # ])
            train_set = torchvision.datasets.CIFAR10(
                root=args.dataset_path, train=True, download=False, transform=transforms.Resize((64, 64)))
            test_set = torchvision.datasets.CIFAR10(
                root=args.dataset_path, train=False, download=False, transform=transforms.Resize((64, 64)))
            dataset = train_set + test_set
            
        elif args.dataset.lower() == 'cifar100':
            from torchvision.datasets import CIFAR100
            train_dataset_without_transform = CIFAR100(
                root=args.dataset_path,
                train=True,
                download=True, transform=transforms.Resize((64, 64)))
            test_dataset_without_transform = CIFAR100(
                root=args.dataset_path,
                train=False,
                download=True, transform=transforms.Resize((64, 64)))
            
        length = len(dataset)
        each_length=length//6
        target_train=torch.utils.data.Subset(dataset,[i for i in range(0,each_length*2)])
        target_test=torch.utils.data.Subset(dataset,[i for i in range(each_length*2,each_length*3)])
        shadow_train=torch.utils.data.Subset(dataset,[i for i in range(each_length*3,each_length*5)])    # for inference attacks
        shadow_test=torch.utils.data.Subset(dataset,[i for i in range(each_length*5,each_length*6)])     # for inference attacks
        train_dataset_without_transform = target_train
        test_dataset_without_transform = target_test
        

    return train_dataset_without_transform, \
           train_img_transform, \
           train_label_transform, \
           test_dataset_without_transform, \
           test_img_transform, \
           test_label_transform
