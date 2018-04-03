# Week 10 Deep Learning

### Convolutional Neural Network
* Neural Style Transfer -- Putting pictures together for style.

###### Convolutions
A convolution is of the following form.
$$\text{Original Matrix} * \text{Filter Matrix} = \text{Resulting Matrix}$$
where * denotes the convolution operation. The convolution is computed as element-wise multiplication and then sum by rolling the filter matrix through the original matrix. If the original matrix is $n\times n$ and filter matrix is of $f\times f$, then the resulting matrix is $(n-f+1)\times (n-f+1)$.

For example, the vertical line detection convolution,

$$
\left(\begin{array}{cccccc}
9 & 9 & 9 & 0 & 0 & 0 \\
9 & 9 & 9 & 0 & 0 & 0 \\
9 & 9 & 9 & 0 & 0 & 0 \\
9 & 9 & 9 & 0 & 0 & 0 \\
9 & 9 & 9 & 0 & 0 & 0
\end{array}\right)
*
\left(\begin{array}{cc}
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1\\
\end{array}\right)
=
\left(\begin{array}{cc}
0 & 27 & 27 & 0\\
0 & 27 & 27 & 0\\
0 & 27 & 27 & 0\\
0 & 27 & 27 & 0\\
\end{array}\right)
$$

Hence, it's rather clear that the original matrix (pixel matrix for image) has a vertical line in the middle. Filter and kernels are different terminologies used by different people, but they are the same thing.

Some more filter matrixes
$$\text{Vertical Edge: }
\left(\begin{array}{cc}
1 & 1 & 1\\
0 & 0 & 0\\
-1 & -1 & -1\\
\end{array}\right)
$$
$$\text{Sobel: }
\left(\begin{array}{cc}
1 & 0 & -1\\
2 & 0 & -2\\
1 & 0 & -1\\
\end{array}\right)
$$

In the era of deep learning, we can learn the filter matrix as parameters of the model. The NN can generally learn this more robustly compared to hand coded filter by humans.


###### Padding
Because the operation over convolution filter reduces the original matrix, and it under-utilize points at the corner of the matrix, we need padding to ameliorate the problem. For example, we can pad the original matrix by additional rows. If we pad the original matrix by two additional rows and columns (one on each side of the matrix), we would get the size of the original matrix back after convolution. By convention, we pad the original matrix with 0s.

Some terminologies:
* "Valid Convolution": no padding
* "Same Convolution": output size is the same as the input matrix.

Generally, we use odd dimension filters.

###### Strided Convolutions
The original convolution is stride = 1. If we want the operation to roll two columns or rows at a time, we use stride = 2 to denote that. If the original matrix is $n\times n$, filter matrix is of $f\times f$, padding p and stride s, then the resulting matrix is $((n + 2p - f)/s + 1)\times ((n + 2p - f)/s + 1)$. If the number is not integer, then round down (floor operation).

In math textbook, we flip the filter upside down and left to right before doing the computation. In computer science, we simplify this step (and mathematician generally calls cross correlation).

###### Convolution over 3D
In 3D convolution, we will create 3D filter matrix. The number of dimension in channels (color channels) need to match the last dimension of the filter matrix. In this way, we get a 2D matrix out after the convolution operation. Hence, if the original matrix is $n\times n \times n_c$ and filter matrix is of $f\times f \times n_c$, then the resulting matrix is $(n-f+1)\times (n-f+1)\times n^{'}_c$ where $n^{'}_c$ is the number of filters we apply to the original matrix. Note, depth and channels are the same thing, and are used interchangeably in the literature.

###### One Layer of Convolution Network
The resulting matrix is added with a bias term $b$, and then transformed by a non-linear function such as ReLU. One big advantage of convolution network is that we only learn filters which can be applied over images of artificially high pixel densities. Hence, it's less prone to overfitting.

Notations:
If layer $l$ is a convolution layer:
$$f^{[l]} = \text{filter size}$$
$$p^{[l]} = \text{padding}$$
$$s^{[l]} = \text{stride}$$
$$\text{Input dimension}: n^{[l-1]}_H×n^{[l-1]}_W×n^{[l-1]}_C$$
$$\text{Output dimension}: n^{[l]}_H×n^{[l]}_W×n^{[l]}_C$$
$$\text{Note}: n^{[l]}_{H/W} = \text{floor}(\frac{n^{[l-1]}_{H/W}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1)$$
$$\text{Number of filters}: n^{[l]}_C$$
$$\text{Each Filter dimension}: f^{[l]}×f^{[l]}×n^{[l-1]}_C$$
$$\text{Activations dimension}: n^{[l]}_H×n^{[l]}_W×n^{[l]}_C$$
$$\text{Weights dimension}: f^{[l]}×f^{[l]}×n^{[l-1]}_C×n^{[l]}_C$$
$$\text{Bias dimension}: n^{[l]}_C → (1, 1, 1, n^{[l]}_C)\text{ in computation}$$

###### Deep Convolution Network
At the last stage, flatten/unroll the matrix to a vector and then feed it into a logistic regression or softmax (multinomial) regression.

They generally have 3 type of layers
* Convolution (Conv)
* Pooling (Pool)
* Fully connected layer (FC)

In a deep convolution of network, the depth/channels generally increase along the network until the last layer which is flatten and feed to a logistic or softmax (multinomial) regression.

###### Pooling Layers
Perform an operation (e.g. max, then called max pooling) on each rolling region (same as convolution) to compute one number for each region (represents the region) which preserve certain features of the data and also simplifies the complexity of the image. We can also use average pooling, but less frequently used with some exceptions. We also generally don't use padding in pooling. There are no-parameters to learn in pooling apart from tuning for hyper-parameters.

Generally, we don't count pooling layer in counting the layer of NN because we have no parameters to learn in pooling layer.

###### Fully Connected Layers
After we flatten the pooling/convolution output, we compute a traditional NN and each layer of that traditional NN is called a fully connected layer until the last output layer.

Generally, we have convolution layers -> pooling layer -> convolution layer -> pooling layer -> … -> flatten -> fully connected layer -> fully connected layer -> … -> output layer.

Generally, along the convolution layers, the channels/depth increases and the dimensionality of the fully connected layer decreases.

###### Pros and Cons over Fully Connected Layers
Pros:
* Parameter sharing -- they share filters
* Sparsity of connections -- each output only depends on a limited number of inputs (only the region the filter operates on)
* Translational invariance (shifting in images)

Cons:
* Not capturing all the relations (not so much in images???)

### Tuning!
We generally just use architecture in the literature, not tuning specifically for your problem!
