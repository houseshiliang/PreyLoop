## --------------------------------------------------------- utkface ---------------------------------------------------------
##                                                         ************ ft ************
# python ./backdoor_defenses/ft_with_DP.py --yaml_path ../backdoor_resource/config/defense/ft/default.yaml --attack badnet --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/ft_with_DP.py --yaml_path ../backdoor_resource/config/defense/ft/default.yaml --attack badnet --dataset utkface --model cnn --epsilon 8

# python ./backdoor_defenses/ft_with_DP.py --yaml_path ../backdoor_resource/config/defense/ft/default.yaml --attack wanet --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/ft_with_DP.py --yaml_path ../backdoor_resource/config/defense/ft/default.yaml --attack wanet --dataset utkface --model cnn --epsilon 8

# python ./backdoor_defenses/ft_with_DP.py --yaml_path ../backdoor_resource/config/defense/ft/default.yaml --attack ssba --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/ft_with_DP.py --yaml_path ../backdoor_resource/config/defense/ft/default.yaml --attack ssba --dataset utkface --model cnn --epsilon 8


##                                                         ************ nad ************
# python ./backdoor_defenses/nad_with_DP.py --yaml_path ../backdoor_resource/config/defense/nad/default.yaml --attack badnet --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/nad_with_DP.py --yaml_path ../backdoor_resource/config/defense/nad/default.yaml --attack badnet --dataset utkface --model cnn --epsilon 8

# python ./backdoor_defenses/nad_with_DP.py --yaml_path ../backdoor_resource/config/defense/nad/default.yaml --attack wanet --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/nad_with_DP.py --yaml_path ../backdoor_resource/config/defense/nad/default.yaml --attack wanet --dataset utkface --model cnn --epsilon 8

# python ./backdoor_defenses/nad_with_DP.py --yaml_path ../backdoor_resource/config/defense/nad/default.yaml --attack ssba --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/nad_with_DP.py --yaml_path ../backdoor_resource/config/defense/nad/default.yaml --attack ssba --dataset utkface --model cnn --epsilon 8

##                                                         ************ nc ************
# python ./backdoor_defenses/nc_with_DP.py --yaml_path ../backdoor_resource/config/defense/nc/default.yaml --attack badnet --dataset utkface --model cnn --epsilon 6
python ./backdoor_defenses/nc_with_DP.py --yaml_path ../backdoor_resource/config/defense/nc/default.yaml --attack badnet --dataset utkface --model cnn --epsilon 8

# python ./backdoor_defenses/nc_with_DP.py --yaml_path ../backdoor_resource/config/defense/nc/default.yaml --attack wanet --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/nc_with_DP.py --yaml_path ../backdoor_resource/config/defense/nc/default.yaml --attack wanet --dataset utkface --model cnn --epsilon 8

# python ./backdoor_defenses/nc_with_DP.py --yaml_path ../backdoor_resource/config/defense/nc/default.yaml --attack ssba --dataset utkface --model cnn --epsilon 6
# python ./backdoor_defenses/nc_with_DP.py --yaml_path ../backdoor_resource/config/defense/nc/default.yaml --attack ssba --dataset utkface --model cnn --epsilon 8
