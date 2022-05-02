# Reproduction of *Understanding Deep Learning Requires Rethinking Generalization*

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





