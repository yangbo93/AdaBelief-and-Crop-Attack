# CI-AB-FGM
This repository contains code to reproduce results from the paper:
**Adversarial example generation with AdaBelief Optimizer and Crop Invariance**

## REQUIREMENTS
- Environment Anaconda
- Python 3.7
- Tensorflow 1.14
- Numpy 1.18.1 
- cv2 4.4.0.42
- scipy 1.4.1

## Method
we propose AdaBelief Iterative Fast Gradient Method (ABI-FGM) and Crop-Invariant attack Method (CIM) to improves the transferability of adversarial examples. 
ABI-FGM and CIM can be readily integrated to build a strong gradient-based attack to further boost the success rates of adversarial examples for black-box attacks. 
Moreover, our method can also be naturally combined with other gradient-based attack methods to build a more robust attack to generate more transferable adversarial examples against the defense models.

### Dataset
We use a subset of ImageNet validation set containing 1000 images, most of which are correctly classified by those models.

### Models
We use the ensemble of seven models in our submission, many of which are adversarially trained models. The models can be downloaded in (https://github.com/tensorflow/models/tree/master/research/slim, https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models).










