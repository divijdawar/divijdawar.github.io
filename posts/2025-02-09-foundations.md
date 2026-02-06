---
title: "Foundations of Language Models"
readtime: "7 min read"
bodyClass: "dark"
description: "Large language models have historically been considered a subset of natural language processing. However, in the past half-decade, we have witnessed unprecedented advances in these models."
---

Large language models have historically been considered a subset of natural language processing. However, in the past half-decade, we have witnessed unprecedented advances in these models. This progress raises the question of whether categorizing LLMs solely within the scope of NLP might be unnecessarily restrictive. The following blog post presents my learnings, insights, and thoughts from reading multiple papers, blog posts and various other pieces of text into building and theoretical understanding of language models.

# 1. Tokenization

Tokenization is the process where a piece of text is broken down into smaller units called tokens. These tokens can be words, sub-words, characters, or even smaller units, depending on the tokenization strategy employed.

LLMs are predictive models. They are trained to predict the next token in a sequence, given all the other tokens that appear in the sequence. To contextualize this a bit, let's consider a simple sequence of words, with the final word missing:

<p class="centered-text"><strong>The quick brown fox jumped over the lazy ___</strong></p>

The task of the LLM here is to predict the missing word in the sequence. The LLM starts by first breaking the sentence in individual words like this,

<p class="centered-text"><strong>['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy']</strong></p>

These IDs map to entries in a token embedding table (which will be discussed in the next chapter). In this example, we have chosen words as our tokens. Alternatively, we could use characters as tokens, which would result in the following vector:

<p class="centered-text"><strong>['T', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k', ...]</strong></p>

Such a language model would be <a href="https://medium.com/towards-data-science/character-level-language-model-1439f5dd87fe" class="styled-link">character-based</a>, as opposed to word-based. As opposed to these techniques, modern LLMs have adopted sub-word tokenization.

## 1.1 Byte-Pair Encoding (BPE)

<a href="https://en.wikipedia.org/wiki/Byte_pair_encoding" class="styled-link">Byte-Pair Encoding</a> is a sub-word tokenization algorithm used in natural language processing (NLP) to efficiently break text into smaller units (tokens). It balances character-based and word-based tokenization, allowing models to handle rare words and out-of-vocabulary terms effectively. Let us assume the data to be encoded is

<p class="centered-text"><strong>aaabdaaabac</strong></p>

The byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, such as "Z". Now there is the following data and replacement table:

<p class="centered-text"><strong>aaabdaaabac → ZbdZac</strong></p>

<p class="centered-text">Z=aa</p>

Then the process is repeated with byte pair "ab", replacing it with "Y":

<p class="centered-text"><strong>ZYdZYac<br>Y=ab<br>Z=aa</strong></p>

To visualize this, I encourage you to play with a <a href="https://tiktokenizer.vercel.app/" class="styled-link">tokenizer</a>.

# 2. Embedding

A word embedding table converts discrete words (or tokens) into continuous numerical vectors. These vectors capture semantic and syntactic relationships between words, allowing LLMs to process language meaningfully.
It is essentially a lookup table that maps each token (from a tokenizer like BPE) to a high-dimensional vector. For example, the vector for the word cat can look something like this:

<p class="centered-text">
[0.0074, 0.0030, -0.0105, 0.0742, 0.0765, -0.0011, 0.0265, 0.0106, 0.0191, 0.0038,<br>
-0.0468, -0.0212, 0.0091, 0.0030, -0.0563, -0.0396, -0.0998, -0.0796, …, 0.0002]
</p>

To understand the embedding table better, let's take a look at an example.

<img class="center-image" src="/images/Embedding.png">

The following is a 2-dimensional lookup table. You may notice similar words like sink, bathtub, kitchen etc being grouped together.
The relative distances between words reflect how often they appear in similar contexts in a dataset. Words that appear in similar contexts are closer together in the embedding space. The color coding represents different semantic categories to make visualization clearer.
When a word has two unrelated meanings, as with bank, linguists call them homonyms. When a word has two closely related meanings, as with magazine, linguists call it polysemy. LLMs are able to represent the same word with different vectors depending on the context in which that word appears.

# 3. Pre-Training

Pre-training has been widely popular since the early days of NLP research. For example, early attempts to pre-train deep learning systems include unsupervised learning for RNNs, deep feedforward neural networks, and others.
Models like BERT <a href="https://arxiv.org/abs/1810.04805" class="styled-link">[Delvin et al., 2019]</a>
and GPT were early breakthroughs in the field.

## 3.1 Unsupervised, Supervised and Self-supervised Pre-training

Unsupervised learning represented one of the early approaches to pre-training. Instead of optimizing for a pre-defined task, models capture a generalized understanding.
Supervised pre-training, particularly in sequence models, involves a two-step process: first, encoding input sequences into vector representations, followed by applying a classification layer to create a comprehensive classification system.
During pre-training, the model learns to map input sequences to outputs based on labeled data supervision. Subsequently, the core sequence model (encoder) can be repurposed as a component within a new model for different tasks.
However, this approach faces limitations as model complexity increases, primarily due to the growing requirement for labeled training data.

The third approach is self-supervised learning. In this approach, a neural network is trained using supervised signals generated by itself, rather than those provided by humans.
This is done by constructing its own training tasks directly from unlabeled data, such as having pseudo labels. Self-supervised learning has proven to be so successful that most current SOTA (State-of-the-art) models are based on this paradigm.

<img class="center-image" src="/images/Pre-training.png">

Additional Information:

Zero-Shot learning (ZSL)

Zero-Shot Learning (ZSL) represents an advanced machine learning paradigm where models demonstrate the capability to recognize and execute tasks without prior exposure during their training phase.
This methodology enables models to generalize their knowledge to novel classes or tasks without requiring specific training examples. The underlying principle involves leveraging knowledge acquired from related tasks or classes to make informed inferences about previously unseen scenarios.

Few-shot learning

Few-Shot Learning (FSL) describes a learning approach where models develop competency using minimal training examples, typically ranging from one to five instances. This methodology focuses on rapid adaptation and generalization to new tasks with limited exposure to training data.
However, a significant limitation of FSL lies in its susceptibility to overfitting, as models may become overly specialized to the small set of training examples.

## 3.2 Adapting Pre-training Models

Sequence Encoding Models.

A sequence encoding model transforms an input sequence of words or tokens into either a fixed-length real-valued vector or a sequence of vectors, creating a mathematical representation of the sequence.
These models typically serve as components within larger machine learning systems rather than functioning as independent applications.

Sequence Generation Models.

A sequence generation model produces an output sequence of tokens conditional on a given input context.
These models utilize probabilistic methods to generate coherent sequences of text, code, or other token-based content.

## 3.3 Decoder-only Pre-training

Decoder-only architecture models focus solely on generating sequences <a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" class="styled-link">[OpenAI]</a>.
They predict the next token based on the context provided by all the previous tokens, making them autoregressive. The below image describes the architecture of a decoder-only model.

<img class="center-image" src="/images/Decoder.png" style="height: 600px;">

Let us define a log-scale cross-entropy loss function L (p<sup>θ</sup><sub>i+1</sub>, p<sup>gold</sup><sub>i+1</sub>) to measure the difference between the model prediction and the true prediction.
Given a sequence of 'm' tokens {x0, x1, ....xm}, the sum of the loss over the positions {0, 1,...., m-1}, is given by

<img class="center-image" src="/images/Loss.png">

p<sup>gold</sup><sub>i+1</sub> is the one-hot representation of x<sub>i+1</sub>
This loss function can be extended to a set of sequences 'D.' In this case, the objective of pre-training is to find the best parameters that minimize the loss on 'D'.

<img class="center-image" src="/images/Loss1.png">

## 3.4 Masked Language Modelling

Masked Language Modelling (MLM) helps to develop an understanding of context and meaning by predicting masked(hidden) tokens in a sentence. Eg:

<p class="centered-text">Original: "Deep learning is transforming AI."</p>

<p class="centered-text">Masked: "Deep learning is [MASK] AI."</p>

MLMs help models learn from both preceding and succeeding words. This helps build a richer understanding and deeper embeddings.
There are different ways to mask tokens. We can randomly select token in a sentence to mask them or we can also select consecutive tokens as shown in
<a href="https://arxiv.org/abs/1910.10683" class="styled-link">[Raffel et al., 2020]</a>.

# Additional Resources

1. Foundations of large language models: <a href="https://arxiv.org/abs/2501.09223" class="styled-link">[Tong et al., 2025]</a>
2. Sequence to Sequence Learning with Neural Networks: <a href="https://arxiv.org/abs/1409.3215" class="styled-link">[Sutskever et al., 2014]</a>
3. BERT: <a href="https://arxiv.org/abs/1810.04805" class="styled-link">[Delvin et al., 2019]</a>
