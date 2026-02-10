# 📚 Chapter 4: Recurrent Neural Networks (RNNs)

## 🌟 Overview

Welcome to **Chapter 4** of the AI Course! Up until now, we have mostly dealt with static data (images, tabular data) where the order didn't matter. In this chapter, we introduce **Sequence Modeling**.

We will learn how to build neural networks that have "memory"—networks that can process time-series data, understand sentences, and even generate their own text. We start from the mathematical foundations of a simple RNN and build up to modern architectures like LSTMs and Attention mechanisms.

---

## 🛠️ Prerequisites

Before starting this chapter, ensure you are comfortable with:

* **PyTorch Tensors & Gradients** (Chapter 1-2)
* **Basic Feed-Forward Networks** (`nn.Linear`)
* **Python Generators** (for data loading)

### 📦 Installation

Ensure your environment is set up:

```bash
pip install torch numpy matplotlib

```

*(Note: Jupyter Notebooks in this folder are optimized for PyTorch CPU/MPS/CUDA automatically)*

---

## 📂 Course Structure

This chapter is divided into **5 Progressive Notebooks**. It is highly recommended to do them in order.

### 📈 Part 1: The Foundations (Continuous Data)

#### [Notebook 4.1: RNN Fundamentals]

* **Task**: Predicting a Sine Wave (Regression).
* **Concept**: An introduction to the "Hidden State". We implement a Vanilla RNN from scratch to understand how information flows through time.
* **Key Learnings**:
* Input Shapes `(Batch, Seq, Feature)`
* The RNN `forward` loop.
* Backpropagation Through Time (BTT).



#### [Notebook 4.2: Solving Short-Term Memory]

* **Task**: The "Echo" Problem (Synthetic Memory Task).
* **Concept**: We prove that Vanilla RNNs fail at long sequences due to the **Vanishing Gradient** problem. We introduce Gated Architectures to fix this.
* **Key Learnings**:
* **LSTM** (Long Short-Term Memory) & Cell States.
* **GRU** (Gated Recurrent Units).
* Visualizing Loss Landscapes (Exploding/Vanishing gradients).



---

### 🗣️ Part 2: Natural Language Processing (Discrete Data)

#### [Notebook 4.3: Text Classification]

* **Task**: Sentiment Analysis (Positive vs. Negative Reviews).
* **Concept**: Moving from numbers (floats) to words (integers). How do we represent categorical words as dense vectors?
* **Key Learnings**:
* **Tokenization** & Vocabulary building.
* **Word Embeddings** (`nn.Embedding`).
* **Packing Sequences** (`pack_padded_sequence`) for variable-length batches.



#### [Notebook 4.4: Text Generation (Creative RNN)]

* **Task**: Generating Shakespearean Text / Code.
* **Concept**: Changing the architecture from **Many-to-One** (Classification) to **Many-to-Many** (Generation).
* **Key Learnings**:
* Character-level RNNs.
* **Temperature Sampling** (Controlling "Creativity" vs. "Coherence").
* Stateful Training (Keeping memory across batches).



---

### 🧠 Part 3: Advanced Architectures

#### [Notebook 4.5: Seq2Seq & Attention]

* **Task**: Machine Translation (English to French) or Chatbot.
* **Concept**: How do we map an input sequence of length  to an output sequence of length ?
* **Key Learnings**:
* **Encoder-Decoder** Architecture.
* **Teacher Forcing** during training.
* **The Attention Mechanism**: Allowing the network to "focus" on specific input words while generating output.



---

## 🧪 Common Issues & Debugging

* **Shape Errors**: The #1 error in RNNs is input shape. Remember that PyTorch RNNs default to `(Seq, Batch, Feature)` unless you set `batch_first=True`.
* **Exploding Gradients**: If your loss becomes `NaN` or spikes massively, try using **Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

```


* **Padding**: When using text, never forget to ignore the `<PAD>` token (usually index 0) in your loss function/embedding layer.
