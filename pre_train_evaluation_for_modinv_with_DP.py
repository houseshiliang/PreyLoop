from demoloader.preactresnet18 import PreActResNet18
from demoloader.dataloader import *
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import tqdm
import PIL.Image as Image
import os
from typing import Any, Callable, List, Optional, Union, Tuple
from opacus.validators import ModuleValidator



def train_PreActResNet18(dataset_name):
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
                dataset_name, attr="race", root = "./data/datasets/"+dataset_name, model_name="preactresnet18")

    print("------dataset_name------:", dataset_name)

    train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        shadow_test, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not ModuleValidator.is_valid(shadow_model):
        shadow_model = ModuleValidator.fix(shadow_model)

    model = shadow_model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 100

    train_losses = []
    test_losses = []
    best_accuracy = 0
    for e in range(epochs):
        running_loss = 0
        running_lens = 0
        model.train()
        for id, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            pre = model(images)
            loss = criterion(pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # running_lens += len(train_loader)
            # print("epoch no = ",e,", running_loss = ",running_loss)
            # print("{}/{}".format(id, len(train_loader)))

        if True:
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1) 
                    equals = top_class == labels.view(top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            accuracy = accuracy / len(test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(
                    model.state_dict(), "./data/results/trained_model/"+dataset_name+"_eval_for_DP.pth")
            print("Epoch: {}/{}..".format(e + 1, epochs))
            print("Training Loss: {:.3f}".format(
                running_loss / len(train_loader)))
            print("Test Loss: {:.3f}".format(test_loss / len(test_loader)))
            print("Accuracy: {:.3f}".format(accuracy))
            print("-----------------------------------------------------")


if __name__ == "__main__":
    # datasets_name=["fmnist", "utkface", "stl10", "cifar10"]
    train_PreActResNet18("fmnist")
    train_PreActResNet18("utkface")
    # train_PreActResNet18("stl10")
    train_PreActResNet18("cifar10")
