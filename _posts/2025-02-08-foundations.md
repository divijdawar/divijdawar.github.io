---
title: Foundations of Language Models
---

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div class="logo">
    <a href="https://divijdawar.github.io/">Divij Dawar ☀️</a>
  </div>
  <nav id="menu">
    <a href="https://divijdawar.github.io/">Posts</a>
    <a href="https://divijdawar.github.io/projects.html">Projects</a>
    <a href="https://divijdawar.github.io/faq.html">FAQ</a>
  </nav>
</div>

# Foundations of Language Models

Data: February 9th, 2025 | Reading time: 7 minutes | Author: Divij Dawar

Large language models have historically been considered a subset of natural language processing. However, in the past half-decade, we have witnessed unprecedented advances in these models. This progress raises the question of whether categorizing LLMs solely within the scope of NLP might be unnecessarily restrictive. The following blog post presents my learnings, insights, and thoughts from reading multiple papers, blog posts, and various other pieces of text into building and theoretical understanding of language models.

## 1. Tokenization  <-- Changed to h2 (##)

Tokenization is the process where a piece of text is broken down into smaller units called tokens. These tokens can be words, sub-words, characters, or even smaller units, depending on the tokenization strategy employed.

LLMs are predictive models. They are trained to predict the next token in a sequence, given all the other tokens that appear in the sequence. To contextualize this a bit, let’s consider a simple sequence of words, with the final word missing:

**The quick brown fox jumped over the lazy ___**  <-- Bold text

The task of the LLM here is to predict the missing word in the sequence. The LLM starts by first breaking the sentence in individual words like this:

**['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy']**  <-- Bold text

These IDs map to entries in a token embedding table (which will be discussed in the next chapter). In this example, we have chosen words as our tokens. Alternatively, we could use characters as tokens, which would result in the following vector:

**['T', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k', ...]**  <-- Bold text