# Week 3 of Deep Learning

### Shallow Neural Network

###### Notations
We use superscript $[i]$ to refer layers, and superscript $(i)$ refers to the rows of the samples. We use $X$ to represent the input features, and we can use $a^{[i]}$ to represent the data at each layer. The subscript is used to represent the node number in the layer. For example, $a_1^{[2]}$ is the first node in the second layer. The first layer or input layer is index by $0$. When counting the number of layers in the neural network, we don't include the input layer. So a neural network with 1 hidden layer is a "2-layer NN".

###### Neural Network Representation and Computation
A Neural network is consist of input layer, output layer, and hidden layers in between them. To compute the first hidden layer, we can use the vectorized representation similar to vectorized regression notation (transposed version).
$$Z = W × X + b$$
Then, we can apply the sigmoid function or the ReLU function. The rows corresponds to the features (nodes), and then the columns represent the training samples.
# Make sure you check the dimensions when you compute NN!!!

###### Link functions -- activation functions

1. Sigmoid Function, a smooth function

2. tanh Function -- a output centered (around 0) version of sigmoid, a smooth function too, but mostly works better than the sigmoid function except for the output layer.
$$g(z) = \frac{e^{z} - e^{-z}}{e^{z}+e^{-z}}$$

For both sigmoid function and tanh function, the gradient becomes small when either z becomes really large or small, which makes optimization slow.

3. Rectified Linear Unit -- ReLU function, a connected piece-wise function, learning rate is fast even for large z.

4. Leaky ReLU -- allow the negative part to be non-zero.

Use either ReLU or leaky ReLU for all hidden layers, just not the output layer if the output layer is binary (use sigmoid function for output layer).

###### Why we need non-linear activation function?
If the activation function is linear, the neural network simply simplifies to a linear combination combination of the input features. Hence, it's useless to have hidden layers at all. If the output layer is a $y\in\mathbb{R}$, then you can use linear function for the output layer, but not for any of the hidden layers.

###### Computing Gradient -- Used for back propagation
For sigmoid
$$g'(z) = g(z) \cdot (1-g(z))$$
For tanh
$$tanh'(z) = 1 - (tanh(z))^2$$
$$ReLU'(z) = 1 \text{ if } z\ge0, 0 \text{ if } z<0$$
We modify the gradient appropriately using chain rule for variables with coefficients.

###### Gradient Descent for NN
Parameters (Ws) need to initialized randomly to deal with label switching problem. Otherwise, the nodes in the hidden layer will become symmetric if updating is deterministic function of gradient. Hence, it would make no sense to have multiple hidden nodes. Often, we use scaled random coefficients to small random values to have higher initial learning rate.

Forward propagation with 1 Hidden layer:
$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = g^{[1]}(Z^{[1]})$$
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = g^{[2]}(Z^{[2]}) = σ(Z^{[2]})$$
----- More Generally
$$Z^{[i]} = W^{[i]} A^{[i-1]} + b^{[i]}$$
$$A^{[i]} = g^{[i]}(Z^{[i]}) = σ(Z^{[i]})$$

Backward Propagation:
$$dZ^{[2]} = A^{[2]} - Y$$
$$dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]'}$$
$$db^{[2]} = \frac{1}{m}np.sum(dZ^{[2]}, axis = 1, keepdims=True)$$
$$dZ^{[1]} = W^{[2]'}dZ^{[2]} .* g^{[1]'}(Z^{[1]})$$
$$dW^{[1]} = \frac{1}{m}dZ^{[1]}X^{'}$$
$$db^{[1]} = \frac{1}{m}np.sum(dZ^{[1]}, axis = 1, keepdims=True)$$
----- More Generally -- not for output layer [??? Uncertain]
$$dZ^{[i]} = W^{[i+1]'}dZ^{[i+1]} .* g^{[i]'}(Z^{[i]})$$
$$dW^{[i]} = \frac{1}{m}dZ^{[i]}A^{[i-1]'}$$
$$db^{[i]} = \frac{1}{m}np.sum(dZ^{[i]}, axis = 1, keepdims=True)$$

Here, np.sum(A, axis=1, keepdims=True), axis refers to the axis that you sum along -- axis=1 sum along column (get row sum), and axis=0 sum along rows (get col sum), and $g^{i}(Z)$ refers to the derivative not transpose.

###### Bugs can happen often. So, write checking codes, and also write simulation codes!
