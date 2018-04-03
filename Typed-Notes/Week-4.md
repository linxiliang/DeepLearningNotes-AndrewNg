# Week 4 of Deep Learning

### Deep Neural Network

###### Notations
$L$ denotes the number of layers, and $n^{[l]}$ denotes the number of nodes in the layer $l$.  So, generally, $L-1$ would be the number of hidden layers.

###### Neural Network Representation and Computation
Computation - Forward Propagation
$$Z^{[l]} = W^{[l]} × A^{[l-1]} + b^{[l]}, \quad l=1,2,...,L$$
$$A^{[l]} = g^{l}(Z^{[l]})$$
Use for-loops to compute the entire forward propagation. Let's look at the dimensionality of each variables.
$$W^{[l]} ∼ (n^{[l]}, n^{[l-1]}), \quad l = 1,2,…,L$$
$$b^{[l]} ∼ (n^{[l]}, 1), \quad l = 1,2,…,L$$
$$Z^{[l]} ∼ (n^{[l]}, m), \quad l = 1,2,…,L$$
$$A^{[l]} ∼ (n^{[l]}, m), \quad l = 0,2,…,L$$

Computation -- Backward Propagation
$$dZ^{[l]} = dA^{[l]} .* g^{[l]'}(Z^{[l]}), \quad l = 1, 2, …, L$$
$$dW^{[l]} = \frac{1}{m} dZ{[l]} A^{[l-1]'}, \quad l = 1, 2, …, L$$
$$db^{[l]} = \frac{1}{m} np.sum(dZ^{[l]}, axis=1, keepdims=True)$$
$$dA^{[l-1]} = W^{[l]'} dZ^{[l]} \quad l = 1, 2, …, L$$
For last layer, if logistic regression
$$dA^{[L]} = -\frac{Y}{A^{[L]}} + \frac{1-Y}{1-A^{[L]}}$$
Dimensionality for backward propagation／gradients
$$dW^{[l]} ∼ (n^{[l]}, n^{[l-1]})$$
$$db^{[l]} ∼ (n^{[l]}, 1)$$

Here $g^{i}(Z)$ refers to the derivative not transpose.

###### Parameters versus Hyperparameters
Parameters: W and b
Hyperparameters: learning rate, # of iterations, # of hidden layers, # of hidden units, # choice of activation function, and later, moment, minibatch size, regularization parameters... The choice of these parameters are important, and determines your model. Choosing these parameters is an empirical exercise, and need trial and error system [Q: cannot we use the learning system to learn about these hyper parameters too?].

###### Training a neural network may be memory intensive since we compute many $Z$s, which are very large matrixes if $m$ is large.

###### How do we determine the number of layers and the number of nodes? Both are hidden/latent variables.

From Circuit theory, there are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute. (The theory is on logical trees -- look at Andrew's notes)
