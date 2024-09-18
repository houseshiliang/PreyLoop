### [ac, rdp, gdp, prv], zcdp

###****************** DP *******************************************
### sigma σ = 8.13 / 4.88 / 3.05
# ------------------- fmnist -------------------
# python train_target_with_DP.py --dataset_name fmnist --model_name cnn --DP_type rdp --noise 1.5 --lr 1.0 --batchsize 512
# python train_target_with_DP.py --dataset_name fmnist --model_name preactresnet18 --DP_type rdp --noise 1.5 --lr 1.0 --batchsize 512
# python train_target_with_DP.py --dataset_name fmnist --model_name vgg19 --DP_type rdp --noise 1.5 --lr 1.0 --batchsize 512

# python train_target_with_DP.py --dataset_name fmnist --model_name cnn --DP_type rdp  --noise 2.0 --lr 2.0 --batchsize 1024
# python train_target_with_DP.py --dataset_name fmnist --model_name vgg19 --DP_type rdp  --noise 2.0 --lr 2.0 --batchsize 1024
# python train_target_with_DP.py --dataset_name fmnist --model_name preactresnet18 --DP_type rdp  --noise 2.0 --lr 2.0 --batchsize 1024


# ### ------------------- utkface -------------------
# python train_target_with_DP.py --dataset_name utkface --model_name cnn --DP_type rdp --noise 4.0 --lr 3.0 --batchsize 2048
# python train_target_with_DP.py --dataset_name utkface --model_name vgg19 --DP_type rdp --noise 4.0 --lr 3.0 --batchsize 2048
# python train_target_with_DP.py --dataset_name utkface --model_name preactresnet18 --DP_type rdp --noise 4.0 --lr 3.0 --batchsize 2048


# ### ------------------- stl10 -------------------
# python train_target_with_DP.py --dataset_name stl10 --model_name cnn  --DP_type rdp --noise 3.0 --lr 3.0 --batchsize 2048
# python train_target_with_DP.py --dataset_name stl10 --model_name vgg19  --DP_type rdp --noise 3.0 --lr 3.0 --batchsize 2048
# python train_target_with_DP.py --dataset_name stl10 --model_name preactresnet18  --DP_type rdp --noise 3.0 --lr 3.0 --batchsize 2048


# ### ------------------- cifar10 -------------------
# python train_target_with_DP.py --dataset_name cifar10 --model_name cnn --DP_type rdp --noise 3.0 --lr 2.0 --batchsize 4096
# python train_target_with_DP.py --dataset_name cifar10 --model_name vgg19 --DP_type rdp --noise 3.0 --lr 2.0 --batchsize 4096
# python train_target_with_DP.py --dataset_name cifar10 --model_name preactresnet18 --DP_type rdp --noise 3.0 --lr 2.0 --batchsize 4096