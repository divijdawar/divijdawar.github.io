---
title: "Mechanistic Interpretability- an Overview"
readtime: "9 min read"
bodyClass: "dark"
description: "In this blog I will try to expand on the field of mechanistic interpretability and why it is important for AI safety and AI alignment."
---

In this blog I will try to expand on the field of mechanistic interpretability and why it is important for AI safety and AI alignment. I will also talk about a few of the key concepts of the field and provide links to resources that I myself have referred to. Moreover, the field is filled with complex jargon that I shall try to simplify.

Mechanistic Interpretability (aka mech interp aka MI) is the field of study of reverse engineering neural networks from the learned weights down to human-interpretable algorithms. A statement

<p class="centered-text">y = x + 5</p>

is meaningless unless one understands what x and y are. Ultimately the meaning of y and x comes from how they're used by operations somewhere in the program. But if you were reverse engineering (like a hacker or security researcher trying to analyze a compiled program), you wouldn't see x or y—just raw memory addresses and operations like ADD 5 TO MEMORY AT ADDRESS 0xABC123.
Now lets apply the same idea to neural networks. Neurons in a neural network are like variables in a program and network parameters are like program instructions, determining how the neurons interact. Now in a computer vision model, a neuron might be detecting cat ears, but it's not explicitly labeled as such. A reverse engineer must discover this meaning by seeing how it behaves.

# Important Concepts

## Circuits

Circuits play a fundamental role in this field because they represent structured groups of neurons that perform specific computations. Circuits help us decompose a model's behaviour into meaningful, interpretable structures. For example: Instead of just knowing that a model predicts "cat" for an image, circuits allow us to see which specific neurons detect fur, whiskers, or ears, thus providing us with a fine-grained understanding. Moreover, if a model makes an incorrect prediction, circuits can reveal the faulty components responsible for errors. Once a circuit is identified, we can investigate what it does. Some common circuit types include edge detectors, texture detectors, algorithmic detectors.

## Local Interpretable Model-Agnostic Explanations (LIME)

Local Interpretable Model-Agnostic Explanations (LIME) is a model-agnostic technique that can be applied to any machine learning model, regardless of its internal structure or algorithm. Model-agnostic interpretability methods analyze the relationship between input features and output predictions without requiring access to the model's internal parameters, such as weights or architecture. LIME constructs a simpler, interpretable model to approximate the behavior of a more complex model for a specific instance. It does so by perturbing the input (e.g., hiding parts of an image or removing words from a sentence) and observing how the model's predictions change. The key idea is that while a simple model may not accurately represent the entire AI system, it can provide a reliable local explanation for an individual prediction. This enables us to identify which features (e.g., words, pixels, or symptoms) were most influential in determining a specific output.

## SHapley Additive exPlanations (SHAP)

SHAP, or SHapley Additive exPlanations, is a model agnostic technique from machine learning that helps explain why a model made a particular prediction by showing how much each input feature contributed to that prediction. It uses ideas from game theory to fairly distribute the "credit" for the prediction among the features, making it easier to understand complex models like neural networks.

For example, if you're using a model to predict house prices, SHAP can tell you whether the number of bedrooms or the location had a bigger impact on the predicted price for a specific house. While SHAP can help explain what inputs matter, it doesn't tell us how the network processes those inputs internally.

## Weight-tying

Weight-tying is a technique in neural networks where different parts of the model share the same set of weights instead of having separate weights for each operation. This means that a single set of parameters is reused multiple times throughout the model.

Advantages of weight-tying:

1. Since weight-tying enforces consistency, it reduces the complexity of weight interactions and makes it easier to track how information flows through the network.
2. Because the same weights are reused, analyzing a single set of tied weights helps us understand multiple parts of the model at once.
3. In transformer models, weight-tying is often used in embedding layers and decoder layers. Since embeddings and output projections share the same weights, it simplifies the study of how words are represented and generated.

## Control Vectors

Control Vectors are a method that can be used to influence how a language model behaves by directly adjusting its internal representations. Let's take an intuitive approach to understanding them. Imagine you're driving a car—except this car is a language model spitting out text. Normally, you just press the gas (give it a prompt) and hope it goes where you want.
But what if you could grab a steering wheel to nudge it left toward "happy" or right toward "grumpy"? Control vectors are that wheel. They're a way to tweak the model's "thoughts" mid-drive by pushing its internal signals in a direction you've figured out—like turning up the "positivity dial" to make "The day is okay" become "The day is awesome!"

We begin by identifying certain patterns in a model's hidden states that correlate with specific behaviors or traits (e.g., positivity, truthfulness, or even specific concepts like "humor"). By identifying and manipulating these patterns, you can "steer" the model's output in a desired direction during inference, without needing to retrain or fine-tune the entire model. You then run the model on pairs of contrasting inputs and capture the activations (hidden states) at a specific layer. We then calculate the difference between the activations.
This difference vector represents the "direction" of the feature in the activation space. Upon adding (or subtracting) the control vector to the activation of a new input at runtime and then scaling it with a coefficient, the model generates an output shifted towards that direction.

## Auto-Encoders

Autoencoders are a type of neural network used primarily for dimensionality reduction, feature extraction, and data denoising. They consist of two main parts:

1. Encoder: Compresses the input into a smaller, dense representation (latent space)
2. Decoder: Reconstructs the original input from this compressed representation

The goal is to minimize the difference between the input and the reconstructed output, while the latent space captures meaningful features of the data while discarding noise.

Sparse Autoencoders apply constraints that force most neurons in the hidden layer to be inactive (close to zero or zero) for a given input, ensuring that only a small subset of neurons fire. By enforcing sparsity, each neuron in the latent space tends to encode distinct, interpretable features. This helps separate different concepts, making it easier to analyze what each neuron represents.
Sparse autoencoders tend to discover linear structures in high-dimensional latent spaces, making it easier to apply linear probing to understand what information is encoded in different layers of a model. Linear probing is a technique where researchers train a simple linear classifier on top of a neural network's internal activations to predict specific properties or concepts. If the linear model achieves high accuracy, this suggests the information about that property is linearly encoded in the activation space.
Sparse autoencoders also help in activation steering, where modifications to a few key neurons can directly influence model behavior in a controlled way.

A neuron is monosemantic if it consistently activates for only one interpretable feature across all contexts. On the other hand, a neuron is polysemantic if it fires for multiple unrelated features or concepts, depending on context. Polysemantic neurons are much harder to interpret since the model compresses information to fit many features into limited neurons. Superposition is the phenomenon where a neural network stores multiple features in the same set of neurons or weights—creating overlapping representations—rather than dedicating separate neurons for each feature.

## Causal scrubbing

Casual scrubbing is a technique used to rigorously test whether a hypothesized explanation of a neural network's behavior is correct. It involves removing or altering parts of the network's computations to check if they are actually necessary for the model's final prediction. If removing a computation does not change the model's behaviour, then that part of the explanation was wrong or redundant. If the model's behavior does change significantly, then the removed component was crucial to the network's decision-making process.

# <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" class="styled-link">In-context Learning and Induction heads</a>

In 2021, researchers from Anthropic (including Tom Henighan, Jacob Steinhardt, and others) published the paper
<a href="https://transformer-circuits.pub/2021/framework/index.html" class="styled-link">"A Mathematical Framework for Transformer Circuits"</a>
where they first characterized induction heads. The discovery happened while they were analyzing the internal functioning of small transformer models trained on next-token prediction.
The researchers observed that certain attention heads in these models developed a specific pattern of behavior during training:

1. They would attend strongly to tokens that appeared earlier in the sequence
2. When they found matching tokens, they would predict what came after those tokens in the earlier context

For example, if a sequence contained "The cat sat on the mat. The cat" - an induction head would notice the second "The cat" matches the first appearance, attend to it, and then predict "sat" as the next token.
The researchers realized this was a form of in-context learning - the model was effectively building an algorithm to copy patterns it had seen earlier in the same context. This capability emerged naturally during training rather than being explicitly programmed.

Induction circuits are small, reusable algorithmic structures that form inside transformer models that are designed to handle repeated patterns in the input. Induction heads are specialized attention heads in transformer models that detect repeating patterns in the input and use them to make predictions about what comes next. Induction circuits can span multiple attention heads, MLPs (multi-layer perceptrons), and residual streams, while induction heads are only a single attention head. Think of an induction head as one specialized worker laying bricks and induction circuits as the entire team (including bricklayers, plumbers, and electricians) building the house (the copying/generalization algorithm).
When training a language model, given at least two layers, a phase change occurs early on during training. During this phase change, the majority of in-context learning ability (as measured by difference in loss between tokens early and late in the sequence) is acquired, and simultaneously induction heads form within the model. Induction heads implement inductive reasoning, and are not memorizing a fixed table of n-gram statistics. The rule [A][B] … [A] → [B] applies regardless of what A and B are.

# Additional Resources

1. <a href="https://distill.pub/2020/circuits/zoom-in/" class="styled-link">Zoom In: An Introduction to Circuits</a>
2. <a href="https://arxiv.org/pdf/1606.03490" class="styled-link">Mythos of Model Interpretability</a>
3. <a href="https://transformer-circuits.pub/2022/toy_model/index.html#motivation" class="styled-link">Toy Models of Superposition</a>
4. <a href="https://transformer-circuits.pub/2022/solu/index.html#section-6-3-4" class="styled-link">Softmax Linear Units</a>
5. <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html" class="styled-link">In-context Learning and Induction heads</a>
