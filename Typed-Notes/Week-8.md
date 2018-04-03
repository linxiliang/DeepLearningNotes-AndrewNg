# Week 8 of Deep Learning

### Machine Learning Strategy

###### Orthogonalization
Adjust orthogonalized components/hyper-parameters.

To make model better in Training:
* Bigger network
* Better optimization algorithm
* …

To make model better in dev set:
* Regularization
* Bigger Training Set
* Adjust some other parameters

To make model better in test set:
* Bigger dev set

To make model better in real world:
* Change dev set
* Change cost function

Generally, we shouldn't use stopping early stopping since it adjusts fit on train and test at the same time.

### Setting up your goal

###### Have a single number evaluation metric
It's best to have one objective number, and try to improve on that metric.

For example, in precision (If predict A, indeed A) and recall (If indeed A, predict A) situation, we have two metrics -- hence, not sure which is better. As such, instead, we use $F_1$ score -- Harmonic mean for Precision and Recall.
$$F_1 = \frac{2}{1/Precision + 1/Recall}$$

###### Satisficing and Optimizing Metric.
We can do constrained optimization. For example, we can optimize accuracy subject a running time threshold.

Also, for example, optimizing the accuracy, but put some threshold on false positives.

###### Setting up training, dev, and test set
Make sure your dev and test set come from the same distribution. They should also be similar to data you are expecting to get in the future. In modern days, if the dataset is very big, we generally set a large percentage of data to training data since deep learning models are data hungry.

###### When to change dev/test sets and metrics?
We should try to tailor the algorithm to customer experience, i.e. what's the impact of misclassification to consumers. Change your dev/test set if you find user use the model for a different distribution of data (make sure dev/test and real usage come from the same distribution).

###### Human level performance
We tend to use human level performance because

* We can get more data from humans
* Gain insight from manual error analysis: Why did a person get this right?
* Better analysis of bias/variance.

Hence, if the gap between human and model is big, then we may want to try to improve on bias part. If small, try to improve on the variance part. Bayes error (often proxied by human error) is the best possible error rate.

* Avoidable bias: the difference between Bayes error and training error
* Variance: The difference between training and dev error.

For structured data, it's possible for machine learning algorithms to surpass human level performance, but it's generally very difficult to surpass human level performance for natural perception tasks.
