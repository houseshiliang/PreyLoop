#****************** without backdoor & without DP *******************************************
# # ------------------- fmnist -------------------
# python Inference_attacks.py --dataset_name fmnist --model_name cnn
# python Inference_attacks.py --dataset_name fmnist --model_name vgg19
# python Inference_attacks.py --dataset_name fmnist --model_name preactresnet18

# # ------------------- utkface -------------------
# python Inference_attacks.py --dataset_name utkface --model_name cnn
# python Inference_attacks.py --dataset_name utkface --model_name vgg19
# python Inference_attacks.py --dataset_name utkface --model_name preactresnet18

# # # ------------------- stl10 -------------------
python Inference_attacks.py --dataset_name stl10 --model_name cnn
# python Inference_attacks.py --dataset_name stl10 --model_name vgg19
# python Inference_attacks.py --dataset_name stl10 --model_name preactresnet18

# # ------------------- cifar10 -------------------
# python Inference_attacks.py --dataset_name cifar10 --model_name cnn
# python Inference_attacks.py --dataset_name cifar10 --model_name vgg19
# python Inference_attacks.py --dataset_name cifar10 --model_name preactresnet18



#****************** with DP-rdp *******************************************
# # ------------------- fmnist -------------------
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name cnn --DP_type rdp  --epsilon 2 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name cnn  --DP_type rdp  --epsilon 4 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name cnn  --DP_type rdp  --epsilon 6 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name cnn  --DP_type rdp  --epsilon 8 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name vgg19 --DP_type rdp --epsilon 2 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name vgg19 --DP_type rdp --epsilon 4 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name vgg19 --DP_type rdp --epsilon 6 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist --model_name vgg19 --DP_type rdp --epsilon 8 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist  --model_name preactresnet18 --DP_type rdp --epsilon 2 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist  --model_name preactresnet18 --DP_type rdp --epsilon 4 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist  --model_name preactresnet18 --DP_type rdp --epsilon 6 --noise 2.0 --lr 2.0 --batchsize 1024
# python Inference_attacks_with_DP.py --dataset_name fmnist  --model_name preactresnet18 --DP_type rdp --epsilon 8 --noise 2.0 --lr 2.0 --batchsize 1024

# # ------------------- utkface -------------------
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name cnn --DP_type rdp --epsilon 2 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name cnn --DP_type rdp --epsilon 4 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name cnn --DP_type rdp --epsilon 6 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name cnn --DP_type rdp --epsilon 8 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name vgg19 --DP_type rdp --epsilon 2 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name vgg19 --DP_type rdp --epsilon 4 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name vgg19 --DP_type rdp --epsilon 6 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name vgg19 --DP_type rdp --epsilon 8 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name preactresnet18 --DP_type rdp --epsilon 2 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name preactresnet18 --DP_type rdp --epsilon 4 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name preactresnet18 --DP_type rdp --epsilon 6 --noise 4.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name utkface --model_name preactresnet18 --DP_type rdp --epsilon 8 --noise 4.0 --lr 3.0 --batchsize 2048

# # # ------------------- stl10 -------------------
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name cnn --DP_type rdp --epsilon 2 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name cnn --DP_type rdp --epsilon 4 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name cnn --DP_type rdp --epsilon 6 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name cnn --DP_type rdp --epsilon 8 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name vgg19 --DP_type rdp --epsilon 2 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name vgg19 --DP_type rdp --epsilon 4 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name vgg19 --DP_type rdp --epsilon 6 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name vgg19 --DP_type rdp --epsilon 8 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name preactresnet18 --DP_type rdp --epsilon 2 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name preactresnet18 --DP_type rdp --epsilon 4 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name preactresnet18 --DP_type rdp --epsilon 6 --noise 3.0 --lr 3.0 --batchsize 2048
# python Inference_attacks_with_DP.py --dataset_name stl10 --model_name preactresnet18 --DP_type rdp --epsilon 8 --noise 3.0 --lr 3.0 --batchsize 2048

# # ------------------- cifar10 -------------------
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name cnn --DP_type rdp --epsilon 2 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name cnn --DP_type rdp --epsilon 4 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name cnn --DP_type rdp --epsilon 6 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name cnn --DP_type rdp --epsilon 8 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name vgg19 --DP_type rdp --epsilon 2 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name vgg19 --DP_type rdp --epsilon 4 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name vgg19 --DP_type rdp --epsilon 6 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name vgg19 --DP_type rdp --epsilon 8 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name preactresnet18 --DP_type rdp --epsilon 2 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name preactresnet18 --DP_type rdp --epsilon 4 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name preactresnet18 --DP_type rdp --epsilon 6 --noise 3.0 --lr 2.0 --batchsize 4096
# python Inference_attacks_with_DP.py --dataset_name cifar10 --model_name preactresnet18 --DP_type rdp --epsilon 8 --noise 3.0 --lr 2.0 --batchsize 4096



#****************** with backdoor *******************************************
# [badnet, blended, ft_trojan, inputaware, lc, lf, lira, sig, ssba_replace, wanet]
# ------------------- fmnist -------------------
# python Inference_attacks.py --dataset_name fmnist --model_name cnn --backdoor badnet
# python Inference_attacks.py --dataset_name fmnist --model_name vgg19 --backdoor badnet
# python Inference_attacks.py --dataset_name fmnist --model_name preactresnet18 --backdoor badnet

# ------------------- utkface -------------------
# python Inference_attacks.py --dataset_name utkface --model_name cnn --backdoor badnet
# python Inference_attacks.py --dataset_name utkface --model_name vgg19 --backdoor badnet
# python Inference_attacks.py --dataset_name utkface --model_name preactresnet18 --backdoor badnet

# ------------------- stl10 -------------------
# python Inference_attacks.py --dataset_name stl10 --model_name cnn --backdoor badnet
# python Inference_attacks.py --dataset_name stl10 --model_name vgg19 --backdoor badnet
# python Inference_attacks.py --dataset_name stl10 --model_name preactresnet18 --backdoor badnet

# ------------------- cifar10 -------------------
# python Inference_attacks.py --dataset_name cifar10 --model_name cnn --backdoor badnet
# python Inference_attacks.py --dataset_name cifar10 --model_name vgg19 --backdoor badnet
# python Inference_attacks.py --dataset_name cifar10 --model_name preactresnet18 --backdoor badnet


