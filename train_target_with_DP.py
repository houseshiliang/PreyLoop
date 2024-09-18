import os
import sys
import datetime
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import time
# from inference_attacks.meminf import *
# from inference_attacks.modinv import *
# from inference_attacks.attrinf import *
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

# get_eps={"ac" : {"fmnist" : {8.13 : 3, 4.88 : 5, 3.05 : 8},
#                 "utkface" : {14.5 : 3, 8.7 : 5, 5.44 : 8},
#                 "stl10" : {18.89 : 3, 11.33 : 5, 7.09 : 8},
#                 "cifar10" : {8.78 : 3, 5.27 : 5, 3.30: 8}
#                 }
#         }         

get_eps={"ac" : {"fmnist" : {8.13 : 3, 4.88 : 5, 1.3 : 8},
                "utkface" : {14.5 : 3, 8.7 : 5, 5.44 : 8},
                "stl10" : {18.89 : 3, 11.33 : 5, 7.09 : 8},
                "cifar10" : {8.78 : 3, 5.27 : 5, 3.30: 8}
                }
        }  


# print("eps: ", get_eps["ac"]["stl10"][11.33])

def train_model(device, train_set, test_set, dataset_name, model, model_name, use_DP, DP_type, noise, norm, delta,lr,batchsize):
    print("<****************** model: " + model_name + " ==== dataset: "+dataset_name+" ******************>")
    print("<************ DP_type : " + DP_type + " ************ noise: " + str(noise) + " ************>")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batchsize, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)

    # according to "use_DP", control whether the model should be wrapped in opacus
    # calculate privacy budget based on "noise"
    model = model_training(train_loader, test_loader, model, device,
                            use_DP, DP_type, noise, norm, delta,lr)

    print("length of model.net.state_dict():   ", len(model.net.state_dict()))

    if use_DP:
        TARGET_PATH = "./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth"
        # TARGET_PATH = "./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(get_eps[DP_type][dataset_name][noise]) + ".pth"
    else:
        TARGET_PATH = "./data/results/trained_model/target_" + dataset_name + "_" + model_name + ".pth"
    acc_train = 0
    acc_test = 0
    best_acc_test=0
    best_acc_train=0
    best_epoch=0
    last_epsilon=0
    epsilon=0
    i=0
    # if not use_DP:
    #     epochs=100
    # else:
    #     epochs=200

    while epsilon < 8.02:
        print("<======================= Epoch " + str(i+1) + " =======================>")
        
        acc_train, epsilon = model.train(i)
        print("target training ", acc_train, epsilon)
        
        acc_test = model.test()
        print("target testing ", acc_test)

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

        i+=1

        if acc_test> best_acc_test:
            best_acc_test = acc_test
            best_acc_train = acc_train
            best_epoch = i+1

        if use_DP:
            if last_epsilon<1 and epsilon>=1:
                model.saveModel("./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" 
                                + str(1) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth")
                pd.DataFrame([acc_train, acc_test, epsilon, i+1]).to_csv("./data/results/train_record/" + dataset_name + "_" 
                                + model_name + "_" + DP_type + "_" + str(1) + "_" + str(noise)+ "_" + str(lr) + "_" + str(norm) +"_" + str(batchsize) + ".csv", index=False, header=False)

            elif last_epsilon<2 and epsilon>=2:
                model.saveModel("./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" 
                                + str(2) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth")
                pd.DataFrame([acc_train, acc_test, epsilon, i+1]).to_csv("./data/results/train_record/" + dataset_name + "_" 
                                + model_name + "_" + DP_type + "_" + str(2) + "_" + str(noise)+ "_" + str(lr) + "_" + str(norm) +"_" + str(batchsize) + ".csv", index=False, header=False)

            elif last_epsilon<4 and epsilon>=4:
                model.saveModel("./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" 
                                + str(4) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth")
                pd.DataFrame([acc_train, acc_test, epsilon, i+1]).to_csv("./data/results/train_record/" + dataset_name + "_" 
                                + model_name + "_" + DP_type + "_" + str(4) + "_" + str(noise)+ "_" + str(lr) + "_" + str(norm) +"_" + str(batchsize) + ".csv", index=False, header=False)

            elif last_epsilon<6 and epsilon>=6:
                model.saveModel("./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" 
                                + str(6) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth")
                pd.DataFrame([acc_train, acc_test, epsilon, i+1]).to_csv("./data/results/train_record/" + dataset_name + "_" 
                                + model_name + "_" + DP_type + "_" + str(6) + "_" + str(noise)+ "_" + str(lr) + "_" + str(norm) +"_" + str(batchsize) + ".csv", index=False, header=False)
                
            elif last_epsilon<8 and epsilon>=8:
                model.saveModel("./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" 
                                + str(8) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth")
                pd.DataFrame([acc_train, acc_test, epsilon, i+1]).to_csv("./data/results/train_record/" + dataset_name + "_" 
                                + model_name + "_" + DP_type + "_" + str(8) + "_" + str(noise)+ "_" + str(lr) + "_" + str(norm) +"_" + str(batchsize) + ".csv", index=False, header=False)
        
        last_epsilon = epsilon
    if not use_DP:
        model.saveModel(TARGET_PATH)
    # else:
    #     pd.DataFrame([acc_train, acc_test, last_epsilon, i+1]).to_csv("./data/results/train_record/" + dataset_name + "_" 
    #                             + model_name + "_" + DP_type + "_" + str(last_epsilon) + "_" + str(noise)+ "_" + str(lr) + "_" + str(norm) +"_" + str(batchsize) + ".csv", index=False, header=False)
        


    print("<****************** model: " + model_name + " ==== dataset: "+dataset_name+" ****************** over ******************>")
    if use_DP:
        print("<************ DP_type : " + DP_type + " ************ noise: " + str(noise) + " ************>")
    print("Saved target model!!!")
    print("Finished training!!!")
    print("best acc >> train-test-epoch: ", best_acc_train, best_acc_test,best_epoch)
    if use_DP:
        return acc_train, acc_test, epsilon, 100
    else:
        return best_acc_train, best_acc_test, -1, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race",
                        help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('-dn', '--dataset_name', type=str, default="fmnist") #utkface, stl10, fmnist, cifar10
    parser.add_argument('-mod', '--model_name', type=str, default="cnn")  # cnn, vgg19, preactresnet18
    parser.add_argument('-ud', '--use_DP', action='store_true',default=True)
    parser.add_argument('-dt', '--DP_type', type=str,  default='rdp')  # ac, rdp, gdp, prv,           zcdp
    parser.add_argument('-ne', '--noise', type=float, default=1.0)  # sigma = noise
    parser.add_argument('-nm', '--norm', type=float, default=0.1)
    parser.add_argument('-d', '--delta', type=float, default=1e-5)
    parser.add_argument('-lr', '--lr', type=float, default=2.0)
    parser.add_argument('-bs', '--batchsize', type=int, default=512)
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
    lr=args.lr
    batchsize=args.batchsize
    root = "./data/datasets/"+dataset_name

    # models_name=["cnn", "vgg19", "preactresnet18"]
    # datasets_name=["fmnist", "utkface", "stl10", "cifar10"]




    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
                dataset_name, attr, root, model_name)

    acc_train, acc_test, epsilon ,best_epoch=train_model(device, target_train, target_test, dataset_name, target_model, model_name, 
                                                            use_DP, DP_type, noise, norm, delta,lr,batchsize)
    # if use_DP:
    #     pd.DataFrame([acc_train, acc_test, epsilon ,best_epoch]).to_csv("./data/results/train_record/" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(noise)+ "_" + str(lr)+ "_" + str(batchsize) + ".csv", index=False,
    #                                     header=False)
    if not use_DP:
        pd.DataFrame([acc_train, acc_test, best_epoch]).to_csv("./data/results/train_record/" + dataset_name + "_" + model_name + ".csv", index=False,
                                        header=False)


if __name__ == "__main__":
    start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    main()
    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print("start time: ", start_time)
    print("end time: ", end_time)
    # python train_target.py --dataset_name  --model_name