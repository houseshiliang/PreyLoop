import os
import sys
import datetime
import torch
import argparse
import torch.nn as nn
import torchvision.models as models


# from demoloader.train import *
# from demoloader.DCGAN import *
# from utils.define_models import *
from demoloader.dataloader import *


def test_model(device, target_test, shadow_test, dataset_name, model, model_name, use_DP, noise, norm, delta, backdoor):
    print("<****************** model: " + model_name + " ==== dataset: "+dataset_name+"******************>")
    target_test_loader = torch.utils.data.DataLoader(
        target_test, batch_size=64, shuffle=True, num_workers=2)
    shadow_test_loader = torch.utils.data.DataLoader(
        shadow_test, batch_size=64, shuffle=True, num_workers=2)
    
    # ------------------------------test target model's performance------------------------------------------
    if backdoor=="none":
        target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name + ".pth"
    else:
        target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name +"_" + backdoor + ".pth"
    print("-------path: ", target_path)

    model = model.to(device)
    model.load_state_dict(torch.load(target_path))
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss1 = 0
    test_accuracy1 = 0
    test_loss2 = 0
    test_accuracy2 = 0
    with torch.no_grad():
        for images, labels in target_test_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss1 += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1) 
            equals = top_class == labels.view(top_class.shape)
            test_accuracy1 += torch.mean(equals.type(torch.FloatTensor))
        test_accuracy1 = test_accuracy1 / len(target_test_loader)
        test_loss1 = test_loss1 / len(target_test_loader)

        for images, labels in shadow_test_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss2 += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1) 
            equals = top_class == labels.view(top_class.shape)
            test_accuracy2 += torch.mean(equals.type(torch.FloatTensor))
    test_accuracy2 = test_accuracy2 / len(shadow_test_loader)
    test_loss2 = test_loss2 / len(shadow_test_loader)
    # return test_accuracy1, test_loss1,test_accuracy2, test_loss2

    # ------------------------------test shadow model's performance------------------------------------------
    if backdoor=="none":
        target_path="./data/results/trained_model/shadow_" + dataset_name + "_" + model_name + ".pth"
    else:
        target_path="./data/results/trained_model/shadow_" + dataset_name + "_" + model_name +"_" + backdoor + ".pth"
    print("-------path: ", target_path)

    model = model.to(device)
    model.load_state_dict(torch.load(target_path))
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss3 = 0
    test_accuracy3 = 0
    test_loss4 = 0
    test_accuracy4 = 0
    with torch.no_grad():
        for images, labels in target_test_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss3 += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1) 
            equals = top_class == labels.view(top_class.shape)
            test_accuracy3 += torch.mean(equals.type(torch.FloatTensor))
        test_accuracy3 = test_accuracy3 / len(target_test_loader)
        test_loss3 = test_loss3 / len(target_test_loader)

        for images, labels in shadow_test_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss4 += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1) 
            equals = top_class == labels.view(top_class.shape)
            test_accuracy4 += torch.mean(equals.type(torch.FloatTensor))
    test_accuracy4 = test_accuracy4 / len(shadow_test_loader)
    test_loss4 = test_loss4 / len(shadow_test_loader)
    return test_accuracy1, test_loss1, test_accuracy2, test_loss2, test_accuracy3, test_loss3, test_accuracy4, test_loss4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race",
                        help="For attrinf, two attributes should be in format x_y e.g. race_gender")

    parser.add_argument('-dn', '--dataset_name', type=str, default='cifar10') #utkface, stl10, fmnist, cifar10
    parser.add_argument('-mod', '--model_name', type=str)  # cnn, vgg, preresnet
    parser.add_argument('-bd', '--backdoor', type=str, default='none')

    parser.add_argument('-tm', '--train_model', action='store_true')
    parser.add_argument('-ts', '--train_shadow', action='store_true')
    parser.add_argument('-te', '--train_eval', action='store_true')
    parser.add_argument('-ud', '--use_DP', action='store_true',)
    parser.add_argument('-ne', '--noise', type=float, default=1.3)
    parser.add_argument('-nm', '--norm', type=float, default=1.5)
    parser.add_argument('-d', '--delta', type=float, default=1e-5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    attr = args.attributes
    if "_" in attr:
        attr = attr.split("_")
    
    use_DP = args.use_DP
    noise = args.noise
    norm = args.norm
    delta = args.delta
    train_shadow = args.train_shadow
    train_eval = args.train_eval
    dataset_name = args.dataset_name
    model_name = args.model_name
    backdoor=args.backdoor
    root = "./data/datasets/"+dataset_name
    # models_name=["cnn", "vgg19", "preactresnet18"]
    # datasets_name=["fmnist", "utkface", "stl10", "cifar10"]

    

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
                dataset_name, attr, root, model_name)

    # test_accuracy1, test_loss1,test_accuracy2, test_loss2
    test_accuracy1, test_loss1, test_accuracy2, test_loss2, test_accuracy3, test_loss3, test_accuracy4, test_loss4=test_model(device, 
                        # target_train+target_test+shadow_train+shadow_test, 
                        shadow_train,
                        shadow_test, dataset_name, target_model, model_name, use_DP, noise, norm, delta, backdoor)
    print("----------- target model's performance -----------")
    print("target_test Loss: {:.3f}".format(test_loss1))
    print("target_test Accuracy: {:.3f}".format(test_accuracy1))
    print("shadow_test Loss: {:.3f}".format(test_loss2))
    print("shadow_test Accuracy: {:.3f}".format(test_accuracy2))

    print("----------- shadow model's performance -----------")
    print("target_test Loss: {:.3f}".format(test_loss3))
    print("target_test Accuracy: {:.3f}".format(test_accuracy3))
    print("shadow_test Loss: {:.3f}".format(test_loss4))
    print("shadow_test Accuracy: {:.3f}".format(test_accuracy4))


if __name__ == "__main__":
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # python test_model_performance.py --dataset_name fmnist --model_name preactresnet18 --backdoor badnet
    # python test_model_performance.py --dataset_name fmnist --model_name vgg19 --backdoor badnet
    # python test_model_performance.py --dataset_name fmnist --model_name cnn --backdoor badnet

    # python test_model_performance.py --dataset_name cifar10 --model_name cnn --backdoor badnet