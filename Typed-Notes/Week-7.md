# Week 7 of Deep Learning

### Tuning

###### Importance of hyper parameters
Learning rate $α$ is generally most important.
Then try a little bit around momentum rate $β$.
We should also try around the mini-batch size. In addition, Andrew sometimes feather around with the number of hidden units. Then, # of layers, and the learning rate decay rate.


For tuning, don't use a grid. We generally do a random sample of hyper-parameter pairs. We can also use coarse to fine sampling design. Can we think about an optimization technique?

###### Sampling at Random
For layers, it may be OK to sample at random along the possible ranges.
However, for something like learning rate $α$ or $β$ it may be better to search over the log scale. In python, we can generate log uniform numbers in the range of 0.0001 and 1 in the following way.
```python
r = -4 * np.random.rand(10)
α = 10 ^ (-r)
```
If the scale is at the high range (0.99 to 0.999)
```python
r = np.random.rand()
β = 1-10**(- r - 1)
```

###### Batch Normalization
Give some intermediate values in NN -- $Z^{(1)}, \dots, Z^{(m)}$, compute
$$μ=\frac{1}{m}∑^{}_{i} Z^{(i)}$$
$$σ^2 = \frac{1}{m}∑^{}_{i}(Z^{(i)}-μ)^2$$
$$Z^{(i)}_{\text{norm}}=\frac{Z^{(i)}-μ}{\sqrt{σ^2+ε}}$$
However, because we sometimes don't want the mean of the hidden units to be 0, and variance of the hidden units as 1. So, instead, we compute,
$$\tilde{Z}^{(i)} = γZ^{(i)}_{\text{norm}}+β$$
where $\alpha$ and $\beta$ are learnable parameters.

We can omit the parameter $b^{[l]}$ since we conduct normalization for each hidden unit. Hence, instead, we learning $W, γ, β$.

It works because of three reasons -- (i) Gives an easier function to optimize over; (2) it limits the effects of change in the early layers on later layers; (3) it produces a somewhat regularization effect since normalization is performed for each mini-batch.

For test sample, we need separate estimates of $μ, σ^2$ to perform prediction. Often, we estimate it using exponentially weighted mini-batch mean and variances.
$$μ_{k} = ρμ_{k-1}+(1-ρ)μ^{\{k\}}$$
$$σ^2_{k} = ρσ^2_{k-1}+(1-ρ){σ^{\{k\}}}^2$$
where $k$ represents a mini-batch.

### Multi-class classification
###### Softmax Regression
Let $C$ denote the number of classes. The number of output layers equal $C$. The softmax regression function is just the multinomial logit function. The decision boundries with no-hidden layers between two classes are linear given the functional form assumption.

The hard-max function is essentially an indicator function (1 for the correct class, and 0 for other classes).

The softmax loss function is the likelihood function for multinomial logit.
$$\mathcal{L}(\hat{y}, y) = -∑^m_{i}∑^C_{j=1}y^{(i)}_j\log{\hat{y}^{(i)}_j}$$

For backward propagation (if you need to implement it from the scratch),
$$dZ^{[L]} = \hat{y} - y$$


### Programming Frameworks
###### Criteria of Choice
1. Ease of Programming (deployment and development)
2. Running speed
3. Truly open (open source with good governance, remain open source for a long time)

###### Tensor Flow
For Tensor Flow, we first compute a computational graph which specifies how computation is to be done. Then, we have to call a Tensor Flow session and run the computation graph.
```python
import numpy as np
import tensorflow as tf

# Setting the cost function
w = tf.Variable(0, dtype = tf.float32)
cost = tf.add(tf.add(w**2, tf.multiply(-10., 2)), 25)

# Set the training method
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Setup the Tensor Flow environment
init = tf.global_variables_initalizer()
session = tf.Session()
session.run(init) # Run the initialization

# Get the value, not run yet
print(sess.run(w))

# One step of Gradient Descent
session.run(train)
print(session.run(w))

# One Thousand Steps
for i in range(1000):
  session.run(train)
print(session.run(w))
```

Getting data into tensor flow with possible changing values
```python
# Setting the cost function
w = tf.Variable(0, dtype = tf.float32)
x = tf.placeholder(tf.float32, [3,1]) # Placeholders
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

# The rest is the same except at runtime
coefficient = np.array([[1.], [-10.], [25.]])
session.run(train, feed_dict = {x:coefficients})
print(session.run(w))
```

One hot encoding is essentially creating dummies based on categorical variables.
