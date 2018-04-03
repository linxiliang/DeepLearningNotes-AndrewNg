# Week 6 of Deep Learning

### Optimization Algorithms

###### Mini-Batch Algorithms
It's devised to deal with large data set. Split the training set into mini-batches. Some notation, let the $t$th batch with batch size $k$ be denoted by
$$X^{\{t\}} → (n_x, k) - \text{dimensional} $$
$$Y^{\{t\}} → (1, k) - \text{dimensional} $$

Within each mini-batch, we still do forward and backward propagation using vectorized operation if possible. A single pass through the entire training set is also called "1-Epoch".

Some names:

k = m, then, called batch gradient descent

k = 1, then, called stochastic gradient descent

Generally, we use a mini-batch size between 1 and m, not too small, not too large. Generally, 64, 128, 256, 512, 1048 ... power of 2 generally works well. Make sure your mini-batch fits in the cache of CPU or GPU to make things faster.


###### Exponentially Weighted Averages
This is just similar to moving averages in Time series with exponential decay.
$$ V_t = βV_{t-1} + (1-β)θ_t $$
Hence, $V_t$ is a close approximation of the exponentially decayed averages over the last $1/(1-β)$ time periods.

Bias correction -- get rid of slow start (initial with 0 problem) adjust for $V_t$:
$$ \frac{V_t}{1-β^T} $$


###### Momentum Gradient Descent
On iteration $t$:

Compute $dW, db$ on current mini-batch
$$V_{dW} = βV_{dW} + (1-β)dW$$
$$V_{db} = βV_{db} + (1-β)db$$
In updating step,
$$W := W - αV_{dW}$$
$$b := b - αV_{db}$$
A common choice of $β$ is 0.9.

###### RMSprop
Root mean square prop -- RMSprop
On iteration $t$:

Compute $dW, db$ on current mini-batch
$$S_{dW} = βS_{dW} + (1-β)dW^2$$
$$S_{db} = βS_{db} + (1-β)db^2$$
In updating step,
$$W := W - α\frac{dW}{\sqrt{S_{dW}+ε}}$$
$$b := b - α\frac{db}{\sqrt{S_{db}+ε}}$$
A common choice of $β$ is 0.9, and ε=10^{-8}. This algorithm will penalize for large oscillation in different directions.

###### Adam Algorithms (Adaptive Moment Estimation)
On iteration $t$:

Compute $dW, db$ on current mini-batch
$$V_{dW} = β_1 V_{dW} + (1-β_1)dW$$
$$V_{db} = β_1 V_{db} + (1-β_1)db$$
$$S_{dW} = β_2 S_{dW} + (1-β_2)dW^2$$
$$S_{db} = β_2 S_{db} + (1-β_2)db^2$$

Then do bias correction:
$$V_{dW}^{Corrected} = \frac{V_{dW}}{1-β_1^t}$$
$$V_{db}^{Corrected} = \frac{V_{db}}{1-β_1^t}$$
$$S_{dW}^{Corrected} = \frac{S_{dW}}{1-β_2^t}$$
$$S_{db}^{Corrected} = \frac{S_{db}}{1-β_2^t}$$

In updating step,
$$W := W - α\frac{V_{dW}^{Corrected}}{\sqrt{S_{dW}^{Corrected}+ε}}$$
$$b := b - α\frac{V_{db}^{Corrected}}{\sqrt{S_{db}^{Corrected}+ε}}$$
$α$ often needs to be tuned. A common choice of $β_1$ is 0.9, and $β_2$ is 0.999. $ε$ is often chosen to be $10^{-8}$.


###### Learning Rate Decay
Set the learning rate to be
$$α := α_0 \frac{1}{1+\text{decay-rate}\times\text{epoch-number}}$$
or
$$α := α_0 0.95^{\text{epoch-number}}$$
or
$$α := α_0 \frac{k}{\text{epoch-number}}$$
There are other variants, and some may even do manual tuning on learning rate.

###### Local Optima
The local optima (gradient zero points) is often saddle points for neural network. It's rather unlikely for your neural network to be stuck on a local optima, but the learning can be slow at saddle points.
