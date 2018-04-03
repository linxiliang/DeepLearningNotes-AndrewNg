# Week 11 Deep Learning

### Convolutional Neural Network Case Studies
* Neural Style Transfer -- Putting pictures together for style.

###### Classical Networks

* LeNet-5 -- ConV 5×5 filter and 1 stride → max/average pool with f=2, s=2 → ConV 5×5 and 1 stride → max/average pool with f=2, s=2 → Flatten to be a vector → first fully connected layer → second fully connected layer → output layer. In total, there are 5 layers (2 ConV, 3 FC) with trainable parameters and hence the name LeNet-5.

* AlexNet -- ConV 11×11 filter and 4 stride → max/average pool with f=3, s=2 → ConV 5×5 and 1 stride → max/average pool with f=3, s=2 → ConV 3×3 and 1 stride → ConV 3×3 and 1 stride → ConV 3×3 and 1 stride (with possible depth/channel reduction) → max/average pool with f=3, s=2 → Flatten to be a vector → first fully connected layer → second fully connected layer → output layer. It's similar to LeNet, but it's much bigger.

* VGG-16 (16 Layers) -- Make all ConV = 3×3 filter, s = 1, same padding, and MAX-POOL = 2×2, s= 2. Architecture: 2 Layers of ConV 64 filters → MAX-POOL → 2 Layers of ConV 128 filters → MAX-POOL → 3 Layers of ConV 256 filters → MAX-POOL  → 3 Layers of ConV 512 filters → MAX-POOL → 3 Layers of ConV 512 filters → MAX-POOL → Flatten → First FC → Second FC → Output Layer (SoftMax/multinomial logit).

[Potentially, I should read the papers related to these algorithms -- should read AlexNet first, then VGG-16, and LeNet-5]


###### ResNets
ResNets allows very deep neural network since it allows skipped connections.

In the main path setting:
$a^{[l]}$ → Linear Transformation: $Z^{[l+1]} = W^{[l+1]a^{[l]}+b^{[l+1]}}$ → RELU/other activation function: $a^{[l+1]}=g(z^{[l+1]})$ → Linear Transformation: $Z^{[l+2]} = W^{[l+2]a^{[l+1]}+b^{[l+2]}}$ → RELU/other activation function: $a^{[l+2]}=g(z^{[l+2]})$…

In a short cut / skipping connection setting:
$a^{[l]}$ → added to $Z^{[l+2]} = W^{[l+2]a^{[l+1]}+b^{[l+2]}}$ → RELU activation function: $a^{[l+2]}=g(z^{[l+2]} + a^{[l]})$… The addition of $a^{[l]}$ makes it a Residual block.

Residual Network: A network with a lot of short cuts/residual blocks. Given the activation function is RELU, the model can put 0 as the weighing matrix and bias if the extra layers don't add more information. As a result, valuable information in early layers can directly transfer to later layers. In a plane network, it's difficult for the network to learn the identity functions. Note, it's useful for us to have the dimensionality of $z^{[l+2]}$ and $a^{[l]}$ matching each other. In the case they don't match, we can pre-multiply $a^{[l]}$ with a new weighing matrix resulting in $W_s a^{[l]}$, which matches the dimensionality of $z^{[l+2]}$.

We use ResNets because:
* The skip-connection makes it easy for the network to learn an identity mapping between the input and the output within the ResNet block.
* Using a skip-connection helps the gradient to backpropagate and thus helps you to train deeper networks

###### 1×1 Convolutions
It's also called network in network. In two dimensions, the filter makes little sense it's essentially just multiplying each element of an input matrix by a number. However, in 3D or higher dimensions, a one by one filter compute the sum of the depth. It adds non-linearity in a layer, and hence allows more complex functional pattern.

###### Inception Block and Network (Also called GoogLeNet)
In an inception layer, we can apply multiple filter sizes or even pooling filter (do need padding to make sure the dimensionality matches) to an input image and allow the computer to figure out which filter to use.

However, due to the high computational cost of convoluting with high dimensional filters, we can apply a 1×1 convolution filter to reduce the depth and then apply a more complex filter which can significantly reduce the computational burden. As long as the the number of 1×1 convolution filters are reasonably large, they won't hurt the performance of the model.

We can also reduce the number of channels in the MAX POOLing part of the model.

After we correctly apply the convolutions, we can then concatenate the outputs of different convolutions together to form a new layer (channel concatenation).

An Inception network is to put all a lot of inception blocks together.  We also generally add predictions at some early inception blocks to make sure the early layers ain't doing so bad at predictions. This operation has some regularization effects on the network to avoid overfitting.

### Practical Advices
Start with Open-Source Implementation, and then try to use transfer learning (download the parameters as well since we want to use that info).

We should also use data augmentation if needed such as mirroring, color shifting (One way to implement color shifting is to use PCA (PCA color augmentation)), random cropping, shearing, local warping etc...

At implementation, we can often have 1 CPU thread dedicated to implementing distortions to mini-batch and CPU/GPU to process the mini-batches. The distortions and CPU/GPU training can often be implemented in parallel.

###### Doing well on benchmarks (Not necessarily in production)
* Ensembling -- similar to random forest (can be a little slow because we need to train multiple NNs)
* Multi-corp at test -- Run classifier on multiple versions of test images and average results.
