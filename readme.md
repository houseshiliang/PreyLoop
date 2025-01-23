**********************  public files  **********************
- /data: data file
- /demoloader: pre-defined models and any sundry

****************  Inference attacks' files  ****************
- /inference_attacks: all inference attack functions
- pre_train_evaluation_for_modinv.py: train a predefined model for model inversion attack
- train_target.py: train target models
- Inference_attakcs.py: carry out specific inference attacks

*****************  Backdoor attacks' files  ****************
- /backdoor_attacks: all backdoor attack functions
- /backdoor_defenses : all defenses for backdoor attacks
- /backdoor_config : all attack and defense config file in yaml
- /resource : pre-trained model (eg. auto-encoder for attack)
- /utils : frequent-use functions and other tools
  - ***only define_models.py is used for inference_attacks
  - aggregate_block : frequent-use blocks in script
  - bd_img_transform : basic perturbation on img
  - bd_label_transform : basic transform on label
  - dataset : script for loading the dataset
  - dataset_preprocess : script for preprocess transforms on dataset 
  - backdoor_generate_pindex.py : some function for generation of poison index 
  - bd_dataset.py : the wrapper of backdoored datasets 
  - trainer_cls.py : some basic functions for classification case

