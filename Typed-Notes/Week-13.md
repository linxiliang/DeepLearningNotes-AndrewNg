# Week 13 Deep Learning

### Face Recognition
* Verification: input image, name/ID, and then output whether the input image is that of a claimed person
* Recognition: has a database of K person, get and input image, and output ID if t he image is any of the K persons (or "non recognized")

###### Face Verification Problem
One-shot learning: learning from one example to recognize the person again. Conv Net doesn't work given the limited sample. Instead, we learn a "similarity" function.
$$d(img1, img2) = \text{degree of difference between images}$$
$$d(img1, img2) = \begin{cases}  ≤ τ \text{ then "same"} \\
   > τ \text{ then "different"} \end{cases}.$$

###### Siamese Network
Input an image, and get a representation of the image as a vector $f(x^{(1)})$. Then, we input another image, and get another representation of the image as a vector $f(x^{(2)})$. Then, we feed the two outputs into the similarity function which could simply be $l_2$ norm.

###### Triplet Loss (defined on 3 images)
We use one image as anchor and call it A (Anchor), a picture of the same person and call it P (Positive), and another image of a different person and call it N (Negative). We want
$$d(A, P) = ||f(A) - f(P)||^2$$
to be small, and
$$d(A, N) = ||f(A) - f(N)||^2$$
to be large. At the bare minimum,
$$d(A, P) = ||f(A) - f(P)||^2 < d(A, N) = ||f(A) - f(N)||^2$$
However, we want to have reasonably large margins, and hence, we want
$$||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + α ≤ 0$$
to make sure the algorithm doesn't just encode everything to 0s to exactly satisfy the inequality (here α is the margin term).

Formally, the loss over a single sample is defined as,
$$\mathcal{L}(A,P,N) = \max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + α, 0).$$
Note, we do need multiple images of the same person in training, but in test stage, we don't need that.

During training, if A,P,N are chosen randomly, then $d(A,P) + α ≤ d(A,N)$ is easily satisfied. Instead, we want to choose triplets that are "hard" to train on.  (FaceNet: A unified embedding for face recognition and clustering).

Instead of triplet loss, we can use a logistic regression in the last stage. For example,
$$\hat{y} = σ(∑^{d_x}_k w_k |f_k(x^{(i)}) - f_k(x^{(j)})|$$
At implementation stage, we can pre-compute the anchor images (employee images) representation. Then, we can the loss compared with new image representation. We technically no-longer need anchor images, but instead we just need image pairs with labels.

### Neural Style Transfer
Let C denote the content image, S denote the style image, and G for the generated image. It's technically not about learning about a new neural network, but to use intermediate layers of trained image recognition networks to output an image.

###### What are deep ConvNets learning? (Zeiler and Fergus, 2013, Visualizing ConvNet)
In visualization, we see what picture patterns highly activate the neurons.

In early layers, the ConvNets only scan a small region of the images and generally detect simple patterns. However, at later layers, the result is a function of a large region of the images and would be able to detect more complex patterns.

###### Cost Function
Paper: A neural algorithm of artistic style

We define a cost function over the generated image G as $J(G)$ which represents how good is the generated image. Mathematically,
$$J(G) = α J_{\text{content}}(C, G) + β J_{\text{content}}(S, G), $$
which measures both similarity in contents and similarity in style.

1. Initiate G randomly
2. Use gradient descent to minimize $J(G)$

For context cost function (usually use neurons in the middle of the network), and we use pre-trained ConvNet (e.g. VGG network). Let $a^{[l](C)}$ and $a^{[l](G)}$ be the activation of layer $l$ on the images. Then, if $a^{[l](C)}$ and $a^{[l](G)}$ are similar, both images have similar content. Hence, we use
$$J_{\text{content}} = \frac{1}{2}||a^{[l](C)} - a^{[l](G)}||^2$$
The $\frac{1}{2}$ is less important and can often be modified. 

For style cost function, we need to first define style. We define style as correlation between activations across channels. The correlation essentially tells us whether two layers (image patterns) tend to occur together or not.

Let's first compute the style matrix. Let $a^{[l]}_{i,j,k}$ denotes activation at $(i,j,k)$. The style matrix $G^{[l]}$ (gram matrix, not to be confused with generated image) then will be of the dimensions $n^{[l]}_c × n^{[l]}_c$. Formally,
$$G^{[l]}_{kk'} = ∑^{n^{[l]}_H}_{i=1}∑^{n^{[l]}_W}_{j=1} a^{[l]}_{ijk} a^{[l]}_{ijk'}.$$

We then compute the style matrix for both S and G. Then, we can compute the style cost as,
$$ J^{[l]}_{\text{style}}(S,G) = \frac{1}{(2n^{[l]}_H n^{[l]}_W n^{[l]}_C)^2} ||G^{[l](S)} - G^{[l](G)}||^2_F$$

We can get better results by using multiple layers.
$$ J_{\text{style}} (S,G) = ∑^L_{l} λ^{[l]} J^{[l]}_{\text{style}} (S,G)$$

Hence, together,
$$J(G) = α J_{\text{content}}(C, G) + β J_{\text{content}}(S, G). $$

###### 1D and 3D Generalizations of Convolution
We simply adjust the dimensions of the filters from 2D to 1D and 3D filters. The dimensionality calculation is the same as 2D convolution case.
