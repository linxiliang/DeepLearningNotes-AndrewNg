# Week 9 Deep Learning

### Machine Learning Strategy

###### Error Analysis
Analyze the errors

* Get ~100 mislabeled dev set examples
* Count up how many are dogs
* Check the percentage of problems in this case.

In parallel,
* We can classify different kind of errors.

###### Cleaning up measurement error
Deep learning algorithms are quite robust to random errors, and low percentage of random errors. It's not robust to systematic errors.

For dev and test set, you can add a column to calculate the percentage of mis-labeled data. To determine whether it's worthwhile to correct the mis-labels, we look at

* Overall dev set error
* Errors due to incorrect labels
* Errors due to other causes

If error percentage due to incorrect labels is high, then, it may be worthwhile, otherwise not. Also, we may need to examine the examples where the algorithm agrees with the actual labels but the labels may not be correct.

Hence, it may be worthwhile to look at the actual data sometimes, because it gives us directions by looking at our mistakes.

###### Mismatched training and dev/test set
Suppose, training and testing are on different distributions.
For example, 200,000 images from web, and 10,000 from mobile app. We should 200,000 + 5,000 in training, 2,500 in dev, 2,500 in test. In this way, we make sure the algorithm is tailored to the usage scenario.

To separate the effects of train/dev and different distributions (data mismatch problem), we can carve out a new training dev set from the training sample. Then, the difference between training-dev and training will be the train/dev difference (if big, variance problem).

Overall,
* Human level/Bayes error
* Training set error (Avoidable Bias)
* Training-dev set error (Variance)
* Dev error (Data Mismatch)
* Test error (Overfitting)

To ameliorate data mismatch,
* Carry out manual error analysis to try to understand difference between training and dev/test sets.
* Make training data more similar; or collect more data similar to dev/test sets -- e.g. artificial data synthesis, but be aware to overfitting to the data synthesis space.

###### Transfer Learning (Often used)
If small data, train only parameters for the last layer. Otherwise, train the parameters for all layers (Pre-trainig -- train the previous model, fine tuning -- tuning the parameters in the new data set). Essentially, we try to transfer the knowledge/parameters from one model (with a lot of data) to another setting (with few data points). We generally need similarity in tasks in similar settings.

###### Multi-task Learning (Not used as often, but used in computer vision a lot)
For example, learning about recognizing multiple objects such as self-driving cars trying to recognize both traffic signs and cars (this can deal with missing labels). It makes sense when
* Training on a set of tasks that could benefit from having shared low level features.
* Usually, amount of data you have for each task is quite similar.
* Can train a big enough neural network to do well on all tasks.

###### End-to-end deep learning
It generally works better than the traditional pipeline approach when we have a lot of data! Sometimes, a pipeline approach can work better than end-to-end deep learning (e.g. face recognition, which break the task into two separate simple tasks -- (1) face detection, (2) image recognition. Both have large of amount data that we can use compared to an end to end approach).

The pros and cons --
Pros:
* Let the data speak
* Less hand-designing of components needed

Cons:
* May need large amount of data
* Excludes potentially useful hand-designed components

The key question: Do you have sufficient data to learn a function of the complexity needed to map $x$ to $y$?
