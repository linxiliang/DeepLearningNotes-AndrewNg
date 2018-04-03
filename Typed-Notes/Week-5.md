# Week 5 of Deep Learning

### Tuning the learning model

###### Expertise in one domain doesn't transfer to other settings

###### Iterate on hyper-parameters and experiment on them.

###### Split random select portions in Big Data
Generally, we want to have more in training data, less in dev/holdout/cross-validation sample and test sample. E.g. 10% of data.

Dealing with mis-matched train/test distribution. The goal is to try your best to make sure dev and test set come from the same data distribution.

###### Bias and Variance Tradeoff in Big Data
Look at train set error and dev set error. If train set error << dev set error, then the model is probably overfitting. If train set error $≈$ dev set, but train set has high error, then model is probably under-fitting (high bias). This is operated on Bayes error is approximately 0.

If we can use a bigger network and get more data, we have tools to drive down both Bias and Variance. However, it's not always true.

### Regularization
###### Add regularization terms to our cost function
###### L2 regularization -- Logistic Regression
$$J(w, b) = cost + \frac{λ}{2m}||w||^{2}_{l_2} $$

###### L1 regularization -- Logistic Regression
$$J(w, b) = cost + \frac{λ}{2m}|w|^{}_{l_1} $$

###### For Neural Network
"Frobenius Norm" regularization -- similar to L2 norm, or weight decay
$$J(W, b) = cost + \frac{λ}{2m}∑^{L}_{l=1}||W^{[l]}||^{2}_{F} $$

The "Frobenius Norm" is given by
$${||A||}_{F} = [{∑}_{i,j} a_{i,j}^{2}]^{1/2}$$

With the regularization term -- "Frobenius Norm", the gradient becomes
$$dW^{[l]} = \frac{1}{m} dZ{[l]} A^{[l-1]'} + \frac{λ}{m}W^{[l]}, \quad l = 1, 2, …, L$$

###### Dropout Regularization
Randomly dropout nodes in each layer with probability $p$, but we need to remember to invert the dropout probability (the dropout probability can be different for different layers) to adjust the outcomes/node (Don't do it like a random forest) by dividing the probability of keeping a node in the layer so that each layer across different samples have the same conditional expectation. Also, the dropout among different training samples are independent. However, in the test sample, we shouldn't do random dropout because we don't want to have random predictions. It works because each nodes can be randomly dropped out, hence, it doesn't want to put all weights on a single input node. Hence, it tries to balance the importance of different features a bit, which results in shrinkage of coefficients.

Remember to apply dropout in both forward and backward propagation but not in the test sample.

###### Other Regularization Method
Add more data: Data Augmentation (flip the picture, randomly crop, randomly change orientation, put random distortions). We gets more training sample at low cost, but we lose independence in samples.

Plotting both training error and dev error (initialize as small random values for coefficients), and stop when dev set error starts to go up. However, this is a mix of reducing training error and dev error because we are terminating the algorithm prematurely.

### Setting up your optimization problem

###### Normalizing inputs
Normalize the features to mean 0 and variances 1. Remember to use the same mean and variance for dev and test data set.

###### Vanishing and Exploding Gradients
In deep neural network, the value of gradients may either vanish or explode because the coefficient matrix get multiplied over each other. For example, if linear activation function, coefficient matrixes are just multiplied over and over. If the element is great than 1, then explode, or if <1, then vanish.

We can deal with this problem by carefully choosing the weights/coeffcients. For example, for a layer $l$ with $n$ features:
$$var(W^{[l]}) = 1/n \text{ or } \underbrace{2/n}_{\text{particular for RELU function}}$$
For tanh function, use $\sqrt{1/n}$.

###### Check the gradients
Make sure to check your analytical gradient with the numerical gradient.
If gradient check fails, turn off dropout!
