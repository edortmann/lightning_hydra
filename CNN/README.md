# Experiments

## doubleTraining_lightning.py

This experiment showcases how to define a single CNN model class with Lightning that can be used to be trained on
multiple different datasets, specifically the Mnist and CIFAR10 dataset. Using Lightning, it is possible to use
different optimizers depending on the dataset. Furthermore, it is possible to customize the training step of the model
depending on the dataset, using the hooks Lightning provides.

## multiclass_lightning.py

This experiment showcases how to define a single CNN model class with Lightning that can be used on multiple datasets
that even require different output sizes, specifically the Mnist, CIFAR10, and CIFAR10 restricted to only cat
and dog images. Again, this is possible due to the available hooks provided by Lightning.

## cnn_weight_decay.py

This experiment analyzes the effects of training a CNN model with different weight decays on a classification task.

The foundation for this experiment is proposition 17 from the paper "A Theory of PAC Learnability of Partial Concept
Classes" (Alon et al., FOCS 2021):

$$\text{For all } \gamma, R > 0: VC(\mathbb{H}_{R, \gamma}) = \Theta \left( \dfrac{R^2}{\gamma^2} \right) \text{ and } LD(\mathbb{H}_{R, \gamma}) = \Theta \left( \dfrac{R^2}{\gamma^2} \right)$$

Essentially, the proposition states that for all margins $\gamma$, and positive constant $R$, for the linear classifiers
with at least margin $\gamma$, the VC dimension and the LD dimension scale inversely proportional with $\gamma^2$. 

So, this means that if the margin increases the complexity of the class of linear classifiers decreases, which should
lead to better generalization. 

The following figure shows the results of training a simple 2-convolution-layer CNN for classification on the Mnist
dataset.

![cnn_vis.png](cnn_vis.png)

In this experiment the margin of the model is represented as the inverted frobenius norm of the model parameters and
generalization is tracked as the difference between train and test accuracy, representing the generalization error.
Training the model with different weight decays serves to get different margins since higher weight decay results in
more regularization, specifically smaller weights which lead to a higher margin.

The results seem to align with the proposition. For higher margins, the train-test accuracy difference
(generalization error) gets smaller.

