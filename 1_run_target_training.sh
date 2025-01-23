#****************** without DP *******************************************
# ------------------- fmnist -------------------
# python train_target.py --dataset_name fmnist --model_name cnn
# python train_target.py --dataset_name fmnist --model_name vgg19
# python train_target.py --dataset_name fmnist --model_name preactresnet18

# ------------------- utkface -------------------
# python train_target.py --dataset_name utkface --model_name cnn
# python train_target.py --dataset_name utkface --model_name vgg19
# python train_target.py --dataset_name utkface --model_name preactresnet18

# ------------------- stl10 -------------------
# python train_target.py --dataset_name stl10 --model_name cnn
# python train_target.py --dataset_name stl10 --model_name vgg19
# python train_target.py --dataset_name stl10 --model_name preactresnet18

# ------------------- cifar10 -------------------
# python train_target.py --dataset_name cifar10 --model_name cnn
python train_target.py --dataset_name cifar10 --model_name vgg19 --gpu 1
# python train_target.py --dataset_name cifar10 --model_name preactresnet18 --gpu 0

