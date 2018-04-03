# Week 16 Deep Learning

### Machine Translation

###### Conditional language model
It's consist of a encoding network and the decoding network. In the encoding network, it encodes the input sentences into feature representations, and then use the feature representations to output a translation through a decoding network. Unlike in language output model, we don't sample at random in translation since we want to the best translation. Hence, instead, we output the most likely sentence (not most likely word)(beam search).

###### Beam Search
Let $B$ denote the beam width.

1. Choose the first word given the inputs. However, instead of choosing 1, we choose $B$ number of words.
2. Given the chosen words, input the chosen words to the decoder network, and choose the word (1 only, 3 in total) most likely to go after each chosen words.
3. Proceed similar to step 2 to output the 3 word in the sequence... Iterate until reach \<EOS>.
4. Finally, keep the sentence with the highest probability.

###### Refinement to Beam Search
Length normalization:
Instead of getting the joint probability of the sentence, we use the an average log probabilities as the selection criteria.

###### Error Analysis
We look at the probabilities of human translation sentence and compare that to the probabilities of RNN output. By look at the probabilities, we can conclude about whether RNN is not good enough (P(human) < P(RNN)) or the Beam search is not good (P(human)>P(RNN)). We can use an error tracking table to find which might be better.

###### Bleu Score (Bilingual evaluation understudy, to be read further)
Suppose there are multiple good translations of the same sentence.

Then, we can use Bleu score to evaluate

###### Attention Model
We use a bidirectional RNN for the encoder. Let $a^{<t>}$ denotes activations for each input $t$, and $α^{<s,t>}$ denotes the weight (attention) output $s$ put on input $t$. The context for output $s$ is obtained as a weighted sum of the inputs.
$$C^{<s>} = ∑^{T_x}_{t} α^{<s,t>} a^{<t>}$$

Each output is also put into the context of next output word.

We learn the parameters governing the weights which is the output from a softmax function using a small NN. The computational cost can be high since the cost increases quadratically.

Computer vision researchers have adapted this framework to learn about image captions.


### Speech Recognition
We first pre-process into a spectral gram.

We can have a similar encoding and decoding network to get transcript. We can also use the attention model...

###### CTC Cost (Connectionist Temporal Classification)
Basic rule: collapse repeated characters not separated by blanks.

###### Trigger word detection
We can have a many-to-many RNN, and have it output 1 whenever a trigger word is said.
