# Asymmetric Gradient Penalty Based on Power Exponential Function for Imbalanced Data Classification (GPPE) 
Authors: Linyong Zhou, Guangcan Ran, Hongwei Tan, and Xiaoyao Xie
## Abstract  
Model bias is a tricky problem in imbalanced data classification. An asymmetric gradient penalty method is proposed based on the power exponential function to alleviate this. The methodology integrates a power exponential function as a moderator into the cross entropy loss of the negative samples, driving the model to focus on ”hesitant” samples while ignoring easy and singular samples. The rationality of the algorithm is explored from the gradient point of view, and it is demonstrated that the approach improves focal loss and asymmetric focal loss. Then, the imbalanced data classification experiments were deployed on MNIST, CIFAR10, CIFAR100, and Caltech101, respectively. For binary classification, datasets with several imbalance ratios constituted by varying the sample size of the majority class or minority class are included in the experiments. In the multi-category classification experiments, we discuss imbalanced datasets with only a single majority category as well as those with several majority categories and also examine step imbalance as well as linear imbalanced datasets. The results reveal that the proposed method exhibits competitiveness on various imbalanced datasets and better robustness on high imbalance ratio datasets. Finally, the approach is deployed on the imbalanced pulsar candidate dataset HTRU, and the state-of-the-art results are yielded.
## 1. Envoronment settings 
   * OS: Ubuntu 18.04.2  
   * GPU: Geforce RTX 2070 
   * Cuda: 11.1, Cudnn: v10.0.130  
   * Python: 3.7.11  
   * Pytorch: 1.6.0   
## 2. Preparation
### 2.1. Data  
We performed binary and multi-category classification experiments on MNIST, CIFAR10, CIFAR100, Clatcah101, and HTRU. Part of the data (binary classificaiton dataset on MNIST) is included in the code, others can be downloaded via the link https://pan.baidu.com/s/16BxWAGcyfe79R7DxnmVO6w (extract code: GPPE). HTRU is raw data and can only be used after it has been preprocessed
## 3. Training
### 3.1. File Description 
The file functions are described as
File | Description
--- | --- 
Train_imbalanced_mnist.py | training binary classification on MNIST
Train_imbalanced_cifar10.py | training binary classification on cIFAR10
Train_imbalanced_cifar100.py.py | training binary classification on cIFAR100
Train_imbalanced_htru.py | training binary classification on HTRU
Train_imbalanced_mnist_multi_class.py | training multi-category classification on MNIST
Train_imbalanced_cifar10_multi_class.py | training multi-category classification on cIFAR10
Train_imbalanced_cifar100_multiclass.py | training multi-category classification on cIFAR100
Train_caltech101.py | training multi-category classification on Caltech101
Loss_Function.py | computing GPPE loss 
Model.py | training models on different dataset     
utils.py | computing metrics
### 3.2. Launch the Training
After downloading the code and corresponding data, execute python Train_xxxx.py to launch the training. It should be noted that the code contains binary dataset on MNIST, so “Train_imbalanced_mnist.py” can be executed directly.
## 4. Acknowledgments
Thank the authors of ASL for providing their code. Our code is widely adapted from their repositories.

