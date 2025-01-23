# ------------------- fmnist -------------------
# -->this
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_fmnist.yaml --dataset fmnist --model cnn --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_fmnist.yaml --dataset fmnist --model vgg19 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 64
# -->this
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_fmnist.yaml --dataset fmnist --model preactresnet18 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 64

# -->this
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_fmnist.yaml --dataset fmnist --model cnn --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 256
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_fmnist.yaml --dataset fmnist --model vgg19 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 64
# -->this
python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_fmnist.yaml --dataset fmnist --model preactresnet18 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 64

# -->this
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_fmnist.yaml --dataset fmnist --model cnn --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 128
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_fmnist.yaml --dataset fmnist --model vgg19 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 64
# -->this
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_fmnist.yaml --dataset fmnist --model preactresnet18 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 64

# ------------------- utkface -------------------
# -->this
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_utkface.yaml --dataset utkface --model cnn --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 512
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_utkface.yaml --dataset utkface --model vgg19 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 64
# -->this
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_utkface.yaml --dataset utkface --model preactresnet18 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 64

# -->this
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_utkface.yaml --dataset utkface --model cnn --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 256
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_utkface.yaml --dataset utkface --model vgg19 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048
# -->this
python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_utkface.yaml --dataset utkface --model preactresnet18 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 32

# -->this
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_utkface.yaml --dataset utkface --model cnn --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 256
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_utkface.yaml --dataset utkface --model vgg19 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 32
# -->this
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_utkface.yaml --dataset utkface --model preactresnet18 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 32

# ------------------- stl10 -------------------
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_stl10.yaml --dataset stl10 --model cnn --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 128
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_stl10.yaml --dataset stl10 --model vgg19 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 64
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_stl10.yaml --dataset stl10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 128
 
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_stl10.yaml --dataset stl10 --model cnn --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_stl10.yaml --dataset stl10 --model vgg19 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_stl10.yaml --dataset stl10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048

# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_stl10.yaml --dataset stl10 --model cnn --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_stl10.yaml --dataset stl10 --model vgg19 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_stl10.yaml --dataset stl10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048

# ------------------- cifar10 -------------------
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_cifar10.yaml --dataset cifar10 --model cnn --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 512
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_cifar10.yaml --dataset cifar10 --model vgg19 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 64
# python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default_DP_cifar10.yaml --dataset cifar10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 128

# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_cifar10.yaml --dataset cifar10 --model cnn --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_cifar10.yaml --dataset cifar10 --model vgg19 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/wanet_with_DP.py --yaml_path ../backdoor_resource/config/attack/wanet/default_DP_cifar10.yaml --dataset cifar10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096

# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_cifar10.yaml --dataset cifar10 --model cnn --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_cifar10.yaml --dataset cifar10 --model vgg19 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/ssba_with_DP.py --yaml_path ../backdoor_resource/config/attack/ssba/default_DP_cifar10.yaml --dataset cifar10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
