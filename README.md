# Reproduction of *Understanding Deep Learning Requires Rethinking Generalization*

This is a repository to reproduce the experimental results in the paper, 
> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *Understanding deep learning requires rethinking generalization*. International Conference on Learning Representations (ICLR), 2017. [[arXiv:1611.03530](https://arxiv.org/abs/1611.03530)].

Members: 

Bohan Wang

Ke Wang 

Xinghai Wang

## Datasets 

1. Cifar10: a standard benchmark in image recognition. Cifar10 contains 60,000 images, each of which depicts a natural scene and is of resolution 32*32. 
	    Each image is annotated in one of the 10 classes. 
2. Cifar100: includes images similar to Cifar10. Cifar100 contains 100 object classes with 600images annotated per class

## Structure
**models.py:** contains the model definition code of small Inception, AlexNet and MLP.

**utils.py:** contains the helper functions for pre-processing training data, training and evaluating models. 

**Reproduce.ipynb** contains the codes to reproduce the results of *Understanding Deep Learning Requires Rethinking Generalization*. Note that only the results of the paper are included in this notebook, the extended experiments in the report are not included.


## Instuctions
You can reproduce Fig.1, Fig.2, Table 1 and Table 4 in the paper, 'Understanding Deep Learning Requires Rethinking Generalization' in ``Reproduce.ipynb``




