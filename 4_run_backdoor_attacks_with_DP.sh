# ------------------- fmnist -------------------
python ./backdoor_attacks/badnet_with_DP.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset fmnist --model cnn --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset fmnist --model vgg19 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset fmnist --model preactresnet18 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512

# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset fmnist --model cnn --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset fmnist --model vgg19 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset fmnist --model preactresnet18 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512

# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset fmnist --model cnn --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset fmnist --model vgg19 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset fmnist --model preactresnet18 --DP_type rdp --noise 1.5 --lr 1.0 --batch_size 512

# ------------------- utkface -------------------
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset utkface --model cnn --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset utkface --model vgg19 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset utkface --model preactresnet18 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048

# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset utkface --model cnn --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset utkface --model vgg19 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset utkface --model preactresnet18 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048

# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset utkface --model cnn --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset utkface --model vgg19 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset utkface --model preactresnet18 --DP_type rdp --noise 4.0 --lr 3.0 --batch_size 2048

# ------------------- stl10 -------------------
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset stl10 --model cnn --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset stl10 --model vgg19 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset stl10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
 
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset stl10 --model cnn --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset stl10 --model vgg19 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset stl10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048

# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset stl10 --model cnn --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset stl10 --model vgg19 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset stl10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 3.0 --batch_size 2048

# ------------------- cifar10 -------------------
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset cifar10 --model cnn --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset cifar10 --model vgg19 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset cifar10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096

# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset cifar10 --model cnn --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset cifar10 --model vgg19 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset cifar10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096

# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset cifar10 --model cnn --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset cifar10 --model vgg19 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset cifar10 --model preactresnet18 --DP_type rdp --noise 3.0 --lr 2.0 --batch_size 4096
