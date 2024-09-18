import os
import sys
import datetime
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
import pandas as pd

from inference_attacks.meminf import *
from inference_attacks.modinv import *
from inference_attacks.attrinf import *
from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *
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


def train_model(device, train_set, test_set, dataset_name, model, model_name, use_DP, DP_type, noise, norm, delta):
    print("<****************** model: " + model_name + " ==== dataset: "+dataset_name+" ******************>")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)

    # according to "use_DP", control whether the model should be wrapped in opacus
    # calculate privacy budget based on "noise"
    model = model_training(train_loader, test_loader, model, device,
                            use_DP, DP_type, noise, norm, delta, 0.01)
    # print("length of 11111111111111model.net.state_dict():   ", len(model.net.state_dict()))


    if use_DP:
        TARGET_PATH = "./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(noise) + ".pth"
    else:
        TARGET_PATH = "./data/results/trained_model/target_" + dataset_name + "_" + model_name + ".pth"
    acc_train = 0
    acc_test = 0
    best_acc_test=0
    best_acc_train=0
    best_epoch=0
    for i in range(100):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        
        print("target training")
        acc_train, epsilon = model.train(i)
        
        print("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

        if acc_test> best_acc_test:
            # model.saveModel(TARGET_PATH)
            best_acc_test=acc_test
            best_acc_train=acc_train
            best_epoch=i+1

    print("<****************** model: " + model_name + " ==== dataset: "+dataset_name+" ****************** over ******************>")
    print("Saved target model!!!")
    print("Finished training!!!")

    return best_acc_train, best_acc_test,best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race",
                        help="For attrinf, two attributes should be in format x_y e.g. race_gender")

    parser.add_argument('-dn', '--dataset_name', type=str, default="cifar10") #utkface, stl10, fmnist, cifar10
    parser.add_argument('-mod', '--model_name', type=str, default="preactresnet18")  # cnn, vgg19, preactresnet18

    parser.add_argument('-at', '--attack_type', type=int, default=0)
    parser.add_argument('-tm', '--train_model', action='store_true')

    parser.add_argument('-ud', '--use_DP', action='store_true',default=False)

    parser.add_argument('-dt', '--DP_type', type=str,  default='rdp')  #rdp, gdp, prv, zcdp, ac
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
    DP_type=args.DP_type
    noise = args.noise
    norm = args.norm
    delta = args.delta
    dataset_name = args.dataset_name
    model_name = args.model_name
    root = "./data/datasets/"+dataset_name

    # models_name=["cnn", "vgg19", "preactresnet18"]
    # datasets_name=["fmnist", "utkface", "stl10", "cifar10"]




    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
                dataset_name, attr, root, model_name)

    best_acc_train, best_acc_test,best_epoch=train_model(device, target_train, target_test, dataset_name, target_model, model_name, 
                                                            use_DP, DP_type, noise, norm, delta)
    if use_DP:
        pd.DataFrame([best_acc_train, best_acc_test,best_epoch]).to_csv("./data/results/train_record/" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(noise) + ".csv", index=False,
                                        header=False)
    else:
        pd.DataFrame([best_acc_train, best_acc_test,best_epoch]).to_csv("./data/results/train_record/" + dataset_name + "_" + model_name + ".csv", index=False,
                                        header=False)


if __name__ == "__main__":
    start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    main()
    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print("start time: ", start_time)
    print("end time: ", end_time)
    # python train_target.py --dataset_name  --model_name