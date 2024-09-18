from .preactresnet18 import PreActResNet18
from .vgg_bn import vgg19_bn
# import torchvision.models as models # vgg19 # no channel
# from .preact_resnet import PreActResNet18  # no channel
from typing import Any, Callable, List, Optional, Union, Tuple
from functools import partial
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn as nn
import os
import torch
import random
import numpy as np
# import pandas
import torchvision
from tqdm import tqdm
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()

class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# class CNN(nn.Module):
#     def __init__(self, input_channel=3, num_classes=10):
#         super(CNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(input_channel, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 64, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 128, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(128*6*6, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


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

        # import pdb;pdb.set_trace()
        # for txt_file in self.files:
        #     txt_file_path = os.path.join(self.processed_path, txt_file)
        #     # print(txt_file_path)
        #     with open(txt_file_path, 'r') as f:
        #         assert f is not None
        #         for i in f:
        #             image_name = i.split('jpg ')[0]
        #             attrs = image_name.split('_')
        #             if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
        #                 continue
        #             self.lines.append(image_name+'jpg')

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


def prepare_dataset(dataset, attr, root, model_name):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(
        dataset, attr=attr, root=root, model_name=model_name)
    length = len(dataset)
    # each_length = length//4
    each_length=length//6
    # target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
    #     dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    target_train=torch.utils.data.Subset(dataset,[i for i in range(0,each_length*2)])
    target_test=torch.utils.data.Subset(dataset,[i for i in range(each_length*2,each_length*3)])
    shadow_train=torch.utils.data.Subset(dataset,[i for i in range(each_length*3,each_length*5)])
    shadow_test=torch.utils.data.Subset(dataset,[i for i in range(each_length*5,each_length*6)])

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model


def get_model_dataset(dataset_name, attr, root, model_name):
    if dataset_name.lower() == "utkface":
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError(
                        "Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)
        input_channel = 3

    elif dataset_name.lower() == "fmnist":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 1

    elif dataset_name.lower() == "stl10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_set = torchvision.datasets.STL10(
            root=root, split='train', transform=transform, download=True)

        test_set = torchvision.datasets.STL10(
            root=root, split='test', transform=transform, download=True)

        dataset = train_set + test_set
        input_channel = 3

    elif dataset_name.lower() == "cifar10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 3

    if isinstance(num_classes, int):
        if model_name == 'cnn':
            target_model = CNN(input_channel=input_channel, num_classes=num_classes)
            # target_model = CNN(num_classes=num_classes)
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes)
            # shadow_model = CNN(num_classes=num_classes)
        elif model_name == 'vgg19':
            target_model = vgg19_bn(input_channel=input_channel, num_classes=num_classes)
            # target_model = models.vgg19(num_classes=num_classes)
            shadow_model = vgg19_bn(input_channel=input_channel, num_classes=num_classes)
            # shadow_model = models.vgg19(num_classes=num_classes)
        elif model_name == 'preactresnet18':
            target_model = PreActResNet18(input_channel=input_channel,num_classes=num_classes)
            # target_model = PreActResNet18(num_classes=num_classes)
            shadow_model = PreActResNet18(input_channel=input_channel,num_classes=num_classes)
            # shadow_model = PreActResNet18(num_classes=num_classes)
    else:
        if model_name == 'cnn':
            target_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
            # target_model = CNN(num_classes=num_classes[0])
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
            # shadow_model = CNN(num_classes=num_classes[0])
        elif model_name == 'vgg19':
            target_model = vgg19_bn(input_channel=input_channel, num_classes=num_classes[0])
            # target_model = models.vgg19(num_classes=num_classes[0])
            shadow_model = vgg19_bn(input_channel=input_channel, num_classes=num_classes[0])
            # shadow_model = models.vgg19(num_classes=num_classes[0])

        elif model_name == 'preactresnet18':
            target_model = PreActResNet18(input_channel=input_channel, num_classes=num_classes[0])
            # target_model = PreActResNet18(num_classes=num_classes[0])
            shadow_model = PreActResNet18(input_channel=input_channel, num_classes=num_classes[0])
            # shadow_model = PreActResNet18(num_classes=num_classes[0])
    return num_classes, dataset, target_model, shadow_model
