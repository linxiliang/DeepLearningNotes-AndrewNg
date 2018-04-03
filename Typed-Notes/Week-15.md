# Week 15 Deep Learning

### Word Embedding

###### Word Representation
We can use learn features (called embeddings) from words -- hence we can represent words using those features (featurized representation, embedded to the feature space).

Steps are as follows:

1. Learn word embedding from large text corpus, or download pre-trained embedding online. (Large amount of data, learn the embeddings)
2. Transfer embedding to new task with smaller training set (say, 100k words). (Smaller amount of data)
3. Optional: Continue to finetune the word embeddings with new data.

The embeddings are similar to face features in face recognition.

###### Properties of Word Embeddings
In analogies, we can measure the similarity in feature representations of two words similar to face validation tasks. For example, we can compare the difference of feature representations of man and woman versus King and Queen. For similarity function, we generally use cosine similarity function,

$$\text{sim}(u,v) = \frac{u'v}{||u|| × ||V||}$$

For example, we want to predict what is similar to King as to woman to man. Let $e_i$ represents the feature representation of word $i$. Then, to learn that word, we would want to maximize the similarity,
$$\arg_w \max \text{sim}(e_w, e_{king} - e_{man} + e_{woman})$$

We can visualize the words by using t-SNE mapping by converting the feature representations (embeddings) to 2D.

The embedding matrix is the features (embeddings on the row), and the words in the dictionary in the column (each word is a column).

###### Learning Word Embeddings
Initialize the a feature representation matrix (embedding matrix) (E) as a parameter matrix. Then, get the representations for each of the proceeding words (or around) (e.g. 5 words) the word you want to predict. Then, we first transform the words into their feature representations and then feed them into a neural network to predict the word (in training, we know the word).

###### Word2Vec Skip Gram Model
Word2Vec Skip Gram Model:
1. Randomly pick a word as context from meaningful words (nouns, verbs)
2. Randomly pick another word nearby as target.
3. Put them together as a training dataset.
4. Initialize feature presentation matrix
5. Get the feature representations of the context
6. Feed into NN to predict the target word

The computational cost is high since we have to add up over exponentials of words representations in dictionary in the multinomial (softmax) layer in output layer. One way to solve it is to use a hierarchical softmax (binary trees).

###### Negative Sampling

1. Randomly pick a word as context from meaningful words (nouns, verbs)
2. Randomly pick another word nearby as target.
3. Put the context and the target word together, and put $y=1$ as a positive sample
4. Put the context and a randomly chosen word from the dictionary together (doesn't need to be uniform), and put $y=0$ as a negative sample
5. Put the context and another randomly chosen word from the dictionary together, and put $y=0$ as a negative sample, and repeat $K$ (e.g. 5-10) times to generate $K$ negative examples.
6. Put together a training sample, and then creating a supervised learning task to predict $y$.

###### GloVe word vectors
Some notation: Let $x_{ij}$ as the number of times $i$ appears in context of $j$.
The algorithm simply minimize,
$$\min ∑^D_i ∑^D_j(θ' e_j - log(x_{ij}))^2$$
If $x_{ij}$ is 0, then simply assign log(x_{ij}) to be 0. The output of $θ_i$, and $e_j$ are symmetric. Hence, we can do an average
$$e^{(final)}_w  = \frac{e_w+θ_w}{2}.$$

###### Sentiment Classification
Simple sentiment classification model:
1. Get the feature representations (embeddings) of the words
2. Get the average of the embeddings / feature presentations.
3. Feed the averages to the NN to predict the sentiments.

RNN for sentiment classification
1. Get the feature representations (embeddings) of the words
2. Feed the feature representations into a many to 1 RNN.

###### Debiasing word embeddings
Word embeddings can reflect gender, ethnicity, age, sexual orientation, and other biases of the text used to train the model. Hence, we should try to debiase it since we don't want to predict these biases.

1. Identify bias direction: e.g. $e_{he} - e_{she}$, $e_{male} - e_{female}$, and get the average of these gender bias
2. Neutralize: for every word that is not definitional (definitional words: e.g. father, mother), project to get rid of bias.
3. Equalize pairs: make sure non-gender words  (doctors) are neutralized.
