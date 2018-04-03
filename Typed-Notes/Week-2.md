# Week 2 of Deep Learning

### Logistic Regression
#### Recognizing Cats
Compute images are stored as pixels in 3 matrixes each for a color.

###### Some Definitions/Notations
$X$ is a matrix of $N_x$ and $M$ dimensional matrix, where $M$ denotes the number of training samples, and $N_x$ denotes the dimensionality of the input features. For Neural Network, we also stack $Y$ into $1\times M$ dimensional matrix.

The sigmoid function $σ(z)=\frac{1}{1+e^{-z}}$ (just logistic function)

###### Logistic Cost Function
Loss function:
$$\mathcal{L}(\hat{y}, y)=-(y\log \hat{y} + (1-y)\log (1-\hat{y}))$$

Cost function: - mean of loss function
$$\mathcal{J}(w,b)=\frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

###### Computation Graphs
$$J(a,b,c)=3(a+bc)$$

Then, the logical steps are computing $u = b⋅c$ , and then compute $v = a+u$, and then finally compute $J=3⋅u$.

###### Computing Derivatives
Computing Derivatives is the backward computation -- chain rule. To get the derivatives with respect to $c$, we first take the derivative with the respect to $J$ (one step backward/back propagation), and then take the derivatives with respect to $u$ (second step backward/back propagation), and then finally with respect to $c$ (third step backward/back propagation). As a notation, we would use $dv$ as the derivative with respect to the final node, and in the case of "$v$", it would be the final function value $J$.

"Forward to compute the function, and backward to compute the derivative"

######
