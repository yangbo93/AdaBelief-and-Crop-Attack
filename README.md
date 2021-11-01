# CI-AB-FGM
This repository contains code to reproduce results from the paper:
**Adversarial example generation with AdaBelief Optimizer and Crop Invariance**. This paper has been preprinted as preprint in arXiv with the according link: https://arxiv.org/abs/2102.03726.

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
We randomly select 1000 images belonging to 1000 categories (i.e., one image per category) from the ImageNet verification set, which were correctly classified by all the testing networks. The download link of the dataset is as follows:  http://ml.cs.tsinghua.edu.cn/~yinpeng/adversarial/dataset.zip. The link to the dataset comes from **MI-FGSM** [1]. You can alternatively use the NIPS 2017 competition official dataset. The download link is https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack/data. Moreover, the ImageNet validation dataset is available at https://image-net.org/index.php.

### Models
We use the ensemble of seven models in our submission, many of which are adversarially trained models. The models can be downloaded in (https://github.com/tensorflow/models/tree/master/research/slim, https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models).

## Acknowledgements
[1] Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., and Li, J. (2018). Boosting adversarial attacks with momentum. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 495 9185–9193.










