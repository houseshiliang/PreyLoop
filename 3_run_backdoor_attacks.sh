# ------------------- fmnist -------------------
python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset fmnist --model cnn
python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset fmnist --model vgg19
python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset fmnist --model preactresnet18

python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset fmnist --model cnn
python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset fmnist --model vgg19
python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset fmnist --model preactresnet18

python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset fmnist --model cnn
python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset fmnist --model vgg19
python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset fmnist --model preactresnet18

# ------------------- utkface -------------------
python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset utkface --model cnn
python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset utkface --model vgg19
python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset utkface --model preactresnet18

python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset utkface --model cnn
python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset utkface --model vgg19
python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset utkface --model preactresnet18

python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset utkface --model cnn
python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset utkface --model vgg19
python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset utkface --model preactresnet18

# ------------------- stl10 -------------------
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset stl10 --model cnn
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset stl10 --model vgg19
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset stl10 --model preactresnet18

# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset stl10 --model cnn
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset stl10 --model vgg19
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset stl10 --model preactresnet18

# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset stl10 --model cnn  
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset stl10 --model vgg19
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset stl10 --model preactresnet18

# ------------------- cifar10 -------------------
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset cifar10 --model cnn
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset cifar10 --model vgg19
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset cifar10 --model preactresnet18

# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset cifar10 --model cnn
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset cifar10 --model vgg19
# python ./backdoor_attacks/wanet.py --yaml_path ../backdoor_resource/config/attack/wanet/default.yaml --dataset cifar10 --model preactresnet18

# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset cifar10 --model cnn
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset cifar10 --model vgg19
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset cifar10 --model preactresnet18










# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset fmnist --model vgg19
# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset utkface --model vgg19
# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset stl10 --model vgg19
# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset cifar10 --model vgg19

# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset fmnist --model cnn
# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset utkface --model cnn
# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset stl10 --model cnn
# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset cifar10 --model cnn



# python ./backdoor_attacks/inputaware.py --yaml_path ../backdoor_resource/config/attack/inputaware/default.yaml --dataset fmnist --model vgg19
# python ./backdoor_attacks/inputaware.py --yaml_path ../backdoor_resource/config/attack/inputaware/default.yaml --dataset utkface --model vgg19
# python ./backdoor_attacks/inputaware.py --yaml_path ../backdoor_resource/config/attack/inputaware/default.yaml --dataset stl10 --model vgg19
# python ./backdoor_attacks/inputaware.py --yaml_path ../backdoor_resource/config/attack/inputaware/default.yaml --dataset cifar10 --model vgg19









# Generate trigger
# python ./resource/badnet/generate_white_square.py --image_size 32 --square_size 3 --distance_to_right 0 --distance_to_bottom 0 --output_path ./resource/badnet/trigger_image.png

# Backdoor training


# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset fmnist --model vgg19
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset utkface --model vgg19
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset stl10 --model vgg19
# python ./backdoor_attacks/badnet.py --yaml_path ../backdoor_resource/config/attack/badnet/default.yaml --dataset cifar10 --model vgg19

# python ./backdoor_attacks/blind.py --yaml_path ../backdoor_resource/config/attack/blind/default.yaml --dataset cifar10 --model vgg19
# python ./backdoor_attacks/ssba.py --yaml_path ../backdoor_resource/config/attack/ssba/default.yaml --dataset cifar10 --model vgg19
