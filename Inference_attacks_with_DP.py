import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy
import datetime
import time

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


def train_DCGAN(device, train_set, dataset_name):
    # if backdoor=="none":
    #     attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name
    # else:
    #     attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name+"_" + backdoor
    attack_path="./data/results/inference_attacks/"+ dataset_name

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2)

    if dataset_name.lower() == 'fmnist':
        D = FashionDiscriminator().eval()
        G = FashionGenerator().eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    print("Starting Training DCGAN...")
    GAN = GAN_training(train_loader, D, G, device)
    for i in range(50):
        print("<==================== Epoch " + str(i+1) + " ====================>")
        GAN.train()

    GAN.saveModel(attack_path + "_discriminator.pth", attack_path + "_generator.pth")



def test_meminf(device, num_classes, 
                target_train, target_test, shadow_train, shadow_test, dataset_name, target_model, shadow_model, 
                model_name, train_shadow, use_DP, DP_type, epsilon, noise, norm, delta, lr, batchsize, backdoor):
    shadow_path="./data/results/trained_model/shadow_" + dataset_name + "_" + model_name + ".pth"
    if backdoor=="none":
        if not use_DP:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name + ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name
        else:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(epsilon) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name + "_" + DP_type + "_" + str(epsilon)
    else:
        if not use_DP:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name +"_" + backdoor + ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name+"_" + backdoor
        else:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name +"_" + backdoor +  "_" + DP_type + "_" + str(epsilon) + ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name+"_" + backdoor + "_" + DP_type + "_" + str(epsilon)
    print(target_path)
    batch_size = 128
    if train_shadow:
        shadow_trainloader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_testloader = torch.utils.data.DataLoader(shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        acc_train, acc_test, overfitting=train_shadow_model(shadow_path, device, shadow_model, shadow_trainloader, shadow_testloader, 
                                                                use_DP, noise, norm, loss, optimizer, delta)
        print("Train Acc: "+ str(acc_train)+ " Test Acc: "+str(acc_test)+" overfitting rate: " +str(overfitting))

    # buliding attack dataset------- for both mode3 and mode0
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
                                                    target_train, target_test, shadow_train, shadow_test, batch_size)
    # attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)

    # --------------------------------------- Model 3 -- WhiteBox Shadow ---------------------------------------
    # for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

    # choosing attack model
    attack_model3 = WhiteBoxAttackModel(num_classes, total)    # Model 2 and 3 whitebox 

    print("=========================== attack_mode3_result: ===========================")
    meminf_res_train3, meminf_res_test3=attack_mode3(target_path, shadow_path, attack_path, device, use_DP,
                 attack_trainloader, attack_testloader, target_model, shadow_model, attack_model3, 1, num_classes)

    # --------------------------------------- Model 0 -- BlackBox  Shadow ---------------------------------------
    # choosing attack model
    attack_model0 = ShadowAttackModel(num_classes)    # Model 0 BlackBox Shadow

    print("====================== attack_mode0_result: ======================")
    meminf_res_train0, meminf_res_test0=attack_mode0(target_path, shadow_path, attack_path, device, use_DP,
                            attack_trainloader, attack_testloader, target_model, shadow_model, attack_model0, 1, num_classes)
    
    print("======== " + dataset_name + "_" + model_name+"_ membership inference attack_mode3_result: ========")
    print("train3: ",meminf_res_train3)
    print("test3: ",meminf_res_test3)

    
    print("======== " + dataset_name + "_" + model_name+"_ membership inference attack_mode0_result: ========")
    print("train0: ",meminf_res_train0)
    print("test0: ",meminf_res_test0)
    # return res_train0,res_test0

    return meminf_res_train3, meminf_res_test3, meminf_res_train0, meminf_res_test0
    


def test_modinv(device, num_classes, target_train, target_model, dataset_name, model_name, use_DP, DP_type, epsilon, noise, norm, delta, lr, batchsize, backdoor):
    if backdoor=="none":
        if not use_DP:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name + ".pth"
            # attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name
        else:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(epsilon) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth"
            # attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name + "_" + DP_type + "_" + str(noise)
    else:
        if not use_DP:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name +"_" + backdoor + ".pth"
            # attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name+"_" + backdoor
        else:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name +"_" + backdoor +  "_" + DP_type + "_" + str(epsilon) + ".pth"
            # attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name+"_" + backdoor + "_" + DP_type + "_" + str(noise)
    attack_path="./data/results/inference_attacks/"+ dataset_name
    size = (1,) + tuple(target_train[0][0].shape)

    # target_model, evaluation_model = load_data(
    #     PATH + "_target.pth", PATH + "_eval.pth", target_model, models.resnet18(num_classes=num_classes))
    if dataset_name=="fmnist":
        input_channel = 1
    else:
        input_channel=3
    if not use_DP:
        target_model, evaluation_model = load_data(
                        target_path,"./data/results/trained_model/"+ dataset_name + "_eval.pth", 
                        target_model, 
                        # PreActResNet18(num_classes=num_classes)
                        PreActResNet18(input_channel=input_channel,num_classes=num_classes),
                        use_DP
                        )
    else:
        target_model, evaluation_model = load_data(
                        target_path,"./data/results/trained_model/"+ dataset_name + "_eval_for_DP.pth", 
                        target_model, 
                        # PreActResNet18(num_classes=num_classes)
                        PreActResNet18(input_channel=input_channel,num_classes=num_classes),
                        use_DP
                        )

    # CCS 15 ---------------------------WhiteBox No Auxiliary ---------------------------------------
    '''
    To evaluate the  quality of the reconstructed sample, we first obtain an average sample 
    from all samples of each target class, then calculate the mean squared error (MSE) 
    between this average sample and the reconstructed sample. Finally, we use the average of
    the MSE values for all target classes as the evaluation metric.
    ***Note that smaller MSE within the same dataset indicates better attack performance.***
    '''
    modinv_ccs = ccs_inversion(target_model, size, num_classes, 1, 3000, 100, 0.001, 0.003, device)
    train_loader = torch.utils.data.DataLoader(target_train, batch_size=1, shuffle=False)
    modinv_res_ccs = modinv_ccs.reverse_mse(train_loader).data.cpu().numpy()  # average MSE value of different classes

    # Secret Revealer CVPR 2020 -------- WhiteBox Shadow ---------------------------------------
    if dataset_name.lower() == 'fmnist':
        D = FashionDiscriminator().eval()
        G = FashionGenerator().eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    PATH_D = attack_path + "_discriminator.pth"
    PATH_G = attack_path + "_generator.pth"

    D, G, iden = prepare_GAN(dataset_name, D, G, PATH_D, PATH_G)
    modinv_res_revealer = revealer_inversion(G, D, target_model, evaluation_model, iden, device) # acc

    print("======== " + dataset_name + "_" + model_name+"_ model inversion_WhiteBox No Auxiliary: ========")
    print("mse: ", modinv_res_ccs)
    print("======== " + dataset_name + "_" + model_name+"_ model inversion_WhiteBox Shadow: ========")
    print("acc: ", modinv_res_revealer)
    return float(modinv_res_ccs), modinv_res_revealer



def test_attrinf(device, num_classes, data_train, data_test, dataset_name, target_model, model_name, use_DP, DP_type, epsilon, noise, norm, delta, lr, batchsize, backdoor):
    if backdoor=="none":
        if not use_DP:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name + ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name
        else:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name + "_" + DP_type + "_" + str(epsilon) + "_" + str(noise)+ "_" + str(lr) + "_" + str(batchsize)+ ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name + "_" + DP_type + "_" + str(epsilon)
    else:
        if not use_DP:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name +"_" + backdoor + ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name+"_" + backdoor
        else:
            target_path="./data/results/trained_model/target_" + dataset_name + "_" + model_name +"_" + backdoor +  "_" + DP_type + "_" + str(epsilon) + ".pth"
            attack_path="./data/results/inference_attacks/"+ dataset_name + "_" + model_name+"_" + backdoor + "_" + DP_type + "_" + str(epsilon)

    # attack_length = int(0.5 * len(data_train))
    # rest = len(data_train) - attack_length

    # attack_train, _ = torch.utils.data.random_split(
    #     data_train, [attack_length, rest])
    # attack_test = data_test

    attack_trainloader = torch.utils.data.DataLoader(
        data_train, batch_size=64, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        data_test, batch_size=64, shuffle=True, num_workers=2)

    image_size = [1] + list(data_train[0][0].shape)
    attrinf_res_train, attrinf_res_test=train_attack_model(target_path, attack_path, num_classes, device, use_DP, target_model, 
                            attack_trainloader, attack_testloader, image_size)
    
    print("======== " + dataset_name + "_" + model_name+"_ attribut inference_result: ========")
    print("train: ",attrinf_res_train)
    print("test: ", attrinf_res_test)

    return attrinf_res_train, attrinf_res_test


def str_to_bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race",
                        help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('-dn', '--dataset_name', type=str, default="utkface") #utkface, stl10, fmnist, cifar10
    parser.add_argument('-mod', '--model_name', type=str, default='cnn')  # cnn, vgg19, preactresnet18
    parser.add_argument('-ts', '--train_shadow', action='store_true',default=False)

    parser.add_argument('-bd', '--backdoor', type=str, default='none')
    parser.add_argument('-ud', '--use_DP', action='store_true',default=True)

    parser.add_argument('-dt', '--DP_type', type=str,  default='rdp')  # ac, rdp, gdp, prv,           zcdp
    parser.add_argument('-ep', '--epsilon', type=int,  default=2)
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
    epsilon=args.epsilon
    noise = args.noise
    norm = args.norm
    delta = args.delta
    train_shadow = args.train_shadow
    # train_eval = args.train_eval
    dataset_name = args.dataset_name
    model_name = args.model_name
    lr=args.lr
    batchsize=args.batchsize
    backdoor=args.backdoor
    root = "./data/datasets/" + dataset_name
    # models_name=["cnn", "vgg19", "preactresnet18"]
    # datasets_name=["fmnist", "utkface", "stl10", "cifar10"]


    train_results=[]
    test_results=[]

    # for model_name in models_name:
    #     for dataset_name in datasets_name:
    print("<************ inference_attacks ************ model: " + model_name + " ************ dataset: "+dataset_name+" ************ backdoor: "+backdoor+ " ************>")
    if use_DP:
        print("<************ DP_type : " + DP_type + " ************ noise: " + str(noise) + " ************>")
    # prepare dataset
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
                dataset_name, attr, root, model_name)
        

    # # # -------------- membership inference --------------
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> membership inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    meminf_res_train3, meminf_res_test3, meminf_res_train0, meminf_res_test0=test_meminf(device, num_classes, 
                target_train, target_test, shadow_train, shadow_test, dataset_name, target_model, shadow_model, 
                model_name, train_shadow, use_DP, DP_type, epsilon, noise, norm, delta, lr, batchsize, backdoor)
    # meminf -- WhiteBox Shadow
    # ***F1, AUC, Acc***
    train_results.append(meminf_res_train3)
    test_results.append(meminf_res_test3)
    # meminf -- BlackBox Shadow
    # ***F1, AUC, Acc***
    train_results.append(meminf_res_train0)
    test_results.append(meminf_res_test0)


    # # -------------- model inversion --------------
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model inversion >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    train_DCGAN(device, shadow_test + shadow_train, dataset_name)
    modinv_res_ccs, modinv_res_revealer=test_modinv(device, num_classes, target_train, target_model, dataset_name, 
                                        model_name, use_DP, DP_type, epsilon, noise, norm, delta, lr,batchsize, backdoor)
    # modinv -- WhiteBox No Auxiliary
    # *** MSE *** 
    test_results.append([-1,-1,modinv_res_ccs])
    # modinv -- WhiteBox Shadow
    # *** Acc ***
    test_results.append([-1,-1,modinv_res_revealer])

    if dataset_name=="utkface":
        # -------------- attribut inference --------------
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> attribut inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        attr="race_gender".split("_")
        # prepare dataset
        num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
            dataset_name, attr, root, model_name)

        attrinf_res_train, attrinf_res_test=test_attrinf(device, num_classes, shadow_train, target_train, dataset_name, 
                                           target_model, model_name, use_DP, DP_type, epsilon, noise, norm, delta, lr, batchsize, backdoor)
        # attrinf -- WhiteBox Shadow
        # *** F1, Acc ***
        train_results.append([-1,attrinf_res_train[0],attrinf_res_train[1]])
        test_results.append([-1,attrinf_res_test[0],attrinf_res_test[1]])

    # F1, AUC, Acc -- meminf -- WhiteBox Shadow 
    # F1, AUC, Acc -- meminf -- BlackBox Shadow
    # -1, F1,  Acc -- attrinf-- WhiteBox Shadow
    print("train results: ",train_results)
    # F1, AUC, Acc -- meminf -- WhiteBox Shadow 
    # F1, AUC, Acc -- meminf -- BlackBox Shadow
    # -1, -1,  MSE -- modinv -- WhiteBox No Auxiliary
    # -1, -1,  Acc -- modinv -- WhiteBox Shadow
    # -1, F1,  Acc -- attrinf-- WhiteBox Shadow
    print("test results: ", test_results)
            
    if backdoor=="none":
        if not use_DP:
            pd.DataFrame(train_results).to_csv("./data/results/inference_final_results/train_results_"+dataset_name+"_"+model_name+".csv", index=False,
                                                header=False)
            pd.DataFrame(test_results).to_csv("./data/results/inference_final_results/test_results_"+dataset_name+"_"+model_name+".csv", index=False,
                                                header=False)
        else:
            pd.DataFrame(train_results).to_csv("./data/results/inference_final_results/train_results_"+dataset_name+"_"+model_name+ "_" + DP_type + "_" + str(epsilon) +".csv", index=False,
                                                header=False)
            pd.DataFrame(test_results).to_csv("./data/results/inference_final_results/test_results_"+dataset_name+"_"+model_name+ "_" + DP_type + "_" + str(epsilon) +".csv", index=False,
                                                header=False)
    else:
        if not use_DP:
            pd.DataFrame(train_results).to_csv("./data/results/inference_final_results/train_results_"+dataset_name+"_"+model_name+"_"+backdoor+".csv", index=False,
                                                header=False)
            pd.DataFrame(test_results).to_csv("./data/results/inference_final_results/test_results_"+dataset_name+"_"+model_name+"_"+backdoor+".csv", index=False,
                                                header=False)
        else:
            pd.DataFrame(train_results).to_csv("./data/results/inference_final_results/train_results_"+dataset_name+"_"+model_name+"_"+backdoor+ "_" + DP_type + "_" + str(epsilon) +".csv", index=False,
                                                header=False)
            pd.DataFrame(test_results).to_csv("./data/results/inference_final_results/test_results_"+dataset_name+"_"+model_name+"_"+backdoor+ "_" + DP_type + "_" + str(epsilon) +".csv", index=False,
                                                header=False)
        

    print("<************ inference_attacks ************ model: " + model_name + " ************ dataset: "+dataset_name+" ************ backdoor: "+backdoor+ " ************ over ************>")
    if use_DP:
        print("<************ DP_type : " + DP_type + " ************ noise: " + str(noise) + " ************>")
if __name__ == "__main__":
    start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    main()
    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print("start time: ", start_time)
    print("end time: ", end_time)

