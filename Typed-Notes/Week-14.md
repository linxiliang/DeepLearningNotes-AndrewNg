# Week 14 Deep Learning

### Sequence Models

###### Notations
We let $x^{<t>}$ to denote the $t$th input sequence, and let $x^{(i)<t>}$ denote the $t$th sequence of the $i$th training sample. We let $T^{(i)}_x$ denote the
sequence length of the $i$th training sample input.

We first create a dictionary of words (a vector). Then, we create a dummy (one-hot) vector indicating the position of the word ($x^{<t>}$ is a vector with 0s for all entries except the position where the word located in the dictionary).

###### Recurrent Neural Network Model
We want to use things learned in early sequences to be potentially applied to later part of the sequence.

In a recurrent network, feed $x^{<1>}$ into a NN, and then predict $y^{<1>}$. Then, the activation values of step 1 will be passed along to the second NN together with the second word in the sequence $x^{<2>}$ and predict $y^{<2>}$. Then... repeat... (Recurrent operation). We generally initialize $a^{<0>}$ to be a zero vector. At each step, we compute the activations and predictions as follows:
$$a^{<t>} = g(W_{aa}a^{<t-1>} + W_{ax} X^{<t>} + b_a)$$
$$\hat{y}^{<t>} = g(W_{y}a^{<t>} + b_y)$$
In simplified notation (block matrix notation),
$$a^{<t>} = g(W_{a} [a^{<t-1>}, X^{<t>}] + b_a)$$
$$W_{a}  = [W_{aa}\quad W_{ax}]$$

The parameters for each step of the NN are shared (the same). This model is quite similar to a ARMA model in econometrics.

The current RNN is one directional flowing through the sequence. We can also have bi-directional RNN which allow later sequences to be passed along to early sequences.

###### Backward Propagation in RNN
First, we define the loss at the position $t$ as
$$\mathcal{L}^{<t>}(\hat{y}^{<t>}, y^{<t>}) = \text{Logistic Loss}(\hat{y}^{<t>}, y^{<t>})$$
Or, some other kind of loss function. With this, we use back-propagation through time and compute the appropriate derivatives.

###### Different types of RNN
If $T_x ≢ T_y$, then the previously described RNN wouldn't work. However, we can easily modify the previously structured RNN to accommodate this. For example, in sentiment analysis, we can let the RNN to output only 1 output at the end (Many (inputs, sequence) to one (one output) architecture). We can also allow RNN to have 1 input but many outputs. In "many-to-many" but the output and input lengths are different, we can have RNN to process data (encoder) only and output many outputs (decoder) without further input.

###### Language Model and Sequence Generation
Speech recognition: given inputs, what's the probability of a particular sentence (sequence of words)?
Corpus: A large body of texts
Tokenize: Breakdown the words, and create vector for each words -- we often add <EOS> at the end of a sentence, and replace unknown words with <UNK>.

Then, we build the RNN model to do speech recognition. Given a sequence of words, we can then predict the probability distribution of the next word.

###### Sampling Novel Sequences
To generate novel sequences, we do sequential sampling, not joint sampling. We end the sentence until we get <EOS> or end at pre-determined length. Also, we can choose to reject <UNK> if we would like.

The RNN framework can also deal with character level model, but can be expensive to train.  

###### Gated Recurrent Unit (GRU)
Deep RNNs can be prone to vanishing gradients given the sequence nature of the model.

Define a new variable, $C$ for memory cell. At time $0$, $C^{<0>} = a^{<0>}$. Then, we update $C^{<t>}$ based on the following algorithm,
* Create a proposal: $\tilde{C}^{<t>} = tanh(W_c [Γ_r * C^{<t-1>}, X^{<t>}] + b_c)$
* Create a filter variable/gate: $Γ_u = σ(W_u [C^{<t-1>}, X^{<t>}] + b_u)$
* Create a relevance filter variable/gate: $Γ_r = σ(W_r [C^{<t-1>}, X^{<t>}] + b_r)$
* Update with $C^{<t>}$ probability $Γ_u$: $C^{<t>} = Γ_r * \tilde{C}^{<t>} + (1-Γ_r) * C^{<t-1>}$
where * is the element-wise multiplication. Given the memory, it allows for long-term dependency in words.

###### Long Short Term Memory (LSTM)
It's a more general version of the GRU.

* Create a proposal: $\tilde{C}^{<t>} = tanh(W_c [C^{<t-1>}, X^{<t>}] + b_c)$
* Create a update filter variable/gate: $Γ_u = σ(W_u [C^{<t-1>}, X^{<t>}] + b_u)$
* Create a forget filter variable/gate: $Γ_f = σ(W_f [C^{<t-1>}, X^{<t>}] + b_f)$
* Create an output filter variable/gate: $Γ_o = σ(W_o [C^{<t-1>}, X^{<t>}] + b_o)$
* Update with $C^{<t>}$ probability $Γ_u$: $C^{<t>} = Γ_o * \tilde{C}^{<t>} + Γ_f * C^{<t-1>}$
* In computing activation: $a^{<t>} = Γ_0 * tanh(C^{<t>})$ where * is the element-wise multiplication. Note,  $C^{<t-1>}$ will affect the gate values in LSTM.

###### Bidirectional RNN (BRNN, cannot be used in real time in standard form)
In Bidirectional RNN, we add additional activation unit which flow backward through the time sequence. The calculation is done as follows:
1. Forward calculation of all forward activations
2. From the end of sequence, calculate all backward activations.
3. Given all the the activations, output the prediction.

The output layer is output as follows:
$$\hat{y}^{<t>} = f(\vec{a}^{<t-1>}, \overset{\leftarrow}{a}^{<t+1>}, x^{<t>})$$

###### Deep RNN
For deep RNN, we include multiple activation layers for each sequence of the input (deep in vertical dimension (not along the sequence)).

### [I still don't understand why the data has to be processed sequentially. Why not process the data, and determine where to look at next?]

###### Here is an idea about RNN (Still need to flash it out)
We process the data in two stages
1. We learn about some relational matrices of works -- a highly sparse matrix.
2. Process a word, and then look for the highly related words.
3. Form a sentence or meaning.
