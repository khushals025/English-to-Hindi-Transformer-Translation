# English to Hindi Transformer Translation

This repository contains code explanation and concepts to know for translating English text to Hindi using a Transformer architecture with attention mechanisms.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Transformer Architecture](#transformer-architecture)
- 

## Introduction

The English to Hindi Transformer Translation project aims to implement a neural machine translation system that can translate English sentences to their corresponding Hindi translations using a Transformer architecture. This project leverages the power of attention mechanisms to improve translation accuracy.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/English-to-Hindi-Transformer-Translation.git
```
```bash
cd English-to-Hindi-Transformer-Translation
```
2. Creating a virtual Environment
   
```bash
conda create -n myenv python=3.8
```
```bash
conda install pytorch=1.8.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch
```
```bash
# Verify the installation
python -c "import torch; print(torch.__version__)"
```


3. Dependencies

```bash
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader 
import math 
```

## Dataset
Samanantar is the largest publicly available parallel corpora collection for Indic languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu. The corpus has 49.6M sentence pairs between English to Indian Languages.
I have used English to Hindi Translation dataset in this repository.
- dataset: <https://ai4bharat.iitm.ac.in/samanantar>

## Preprocessing
- ### Tokenization 


Vocabularies are used to initialize Index-to-token and token-to-index mappings for English and Hindi vocabularies, allowing efficient conversion between words and their corresponding numerical indices.
```bash
START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

# Vocabularies
hindi_vocab = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', '।', '॥',
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
    'क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ॐ',
    '०', '१', '२', '३', '४', '५', '६', '७', '८', '९', PADDING_TOKEN, END_TOKEN]

english_vocab = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

```
```bash
index_eng = { i:c for i,c in enumerate(english_vocab)}
eng_index = { c:i for i,c in enumerate(english_vocab)}
index_hin = { i:c for i,c in enumerate(hindi_vocab)}
hin_index = { c:i for i,c in enumerate(hindi_vocab)}
```

- ### Positional Encoding


<div align="center">
  <img src="https://i.stack.imgur.com/E1aEA.jpg" alt="Image Alt Text" width="400">
</div>


The Transformer model does not use recurrence or convolution and treats each data point as independent of the other. Hence, positional information is added to model explicitly retain information regarding other words in a sentence.

Positional Encoding describes the location or position of an entity in a sequence so that each position is assigned a unique representation.


Without positional encoding, an attention only model might believe the following two sentences have the same semantics.
```bash
Tom bit a dog.
A dog bit Tom.
```

- ### How to Encode word positions?


We use sin and cosine functions of different frequencies.We represent each token as a vector of 512 dimensions (from element 0 to element 511). A position (pos) is an integer from 0 to a pre-defined maximum number of tokens in a sentence (a.k.a. max length) minus 1. For example, if the max length is 128, positions are 0 to 127.
Since we have 512 dimensions, we have 256 pairs of sine and cosine values. As such, the value of i goes from 0 to 255.

<div align="center">
  <img src="https://i.stack.imgur.com/67ADh.png" alt="Image Alt Text" width="400">
</div>

Here, d_model = 512
pos is integer within range 0 <= pos < T, where T is input sequence size or max_sequence_len


- ### Sentence Embedding

- first we convert a sentence(string) into list of tokens(values(int) or index of unique vocabularies) using Index-to-token and token-to-index mappings.
- add START_TOKEN at the beginning 
- append END_TOKEN at the end of the sequence
- pad the remaining length ( max_sequence_length - len(sentence) ) with PADDING_TOKEN
- Then compute Positional Encoding 
- And finally add Embeddings with Positional Encodings 

```bash
def forward(self, x , start_token, end_token):
        x = self.batch_tokenize(x , start_token, end_token)
        x = self.embedding(x)
        pos = self.positional_encoder()#.to(get_device())
        x = self.dropout(x + pos)
        return x
    
```


<div align="center">
  <img src="https://kikaben.com/transformers-positional-encoding/images/pos-encoding-plus-word-embedding.png" alt="Image Alt Text" width="400">
</div>


## Transformer Architecture
- Scaled Dot Product Attention
- Multi-Head Self Attention
- Multi-Head Cross Attention
- Layer Normalization
- Feed Forward Neural Network
- Encoder
- Decoder

- ### Scaled Dot Product Attention

<div align="center">
  <img src="https://blog.paperspace.com/content/images/2020/05/1_9nUzdaTbKzJrAsq1qqJNNA.png" alt="Image Alt Text" width="600">
</div>


- Q, K and V are Query , Key and Value with dimensions d_q, = d_k and d_v.
- They are fed into the following formula to compute scaled dot product attention.

<div align="center">
  <img src="https://static.packt-cdn.com/products/9781800565791/graphics/Images/B16681_01_005.png" alt="Image Alt Text" width="350">
</div>



- size of each tensor is (batch_size, sequence_length, d_model)
- where, batch_size = 30, max_sequnece_len = 360, d_model = 512
- We can also initialize Q and K with the following


```bash
from torch import nn 

nn.Linear(d_model, d_model)
```


- or can split or chunk them as I have done in the code 


```bash
self.qkv_layer = nn.Linear(d_model, 3*d_model) # querey, key and value 
.
.
.
q, k, v = qkv.chunk(3, dim=-1) # split into 3 

```

- You may note that the scaled dot-product attention can also apply a mask to the attention scores before feeding them into the softmax function. 
- a look-ahead mask is also required to prevent the decoder from attending to succeeding words, such that the prediction for a particular word can only depend on known outputs for the words that come before it.
These look-ahead and padding masks are applied inside the scaled dot-product attention set to -
∞
all the values in the input to the softmax function that should not be considered. For each of these large negative inputs, the softmax function will, in turn, produce an output value that is close to zero, effectively masking them out.



<div align="center">
  <img src="https://cdn.hashnode.com/res/hashnode/image/upload/v1638824585791/vkXCmdGyw.png?auto=compress,format&format=webp" alt="Image Alt Text" width="500">
</div>



```bash
NEG_INFTY = -1e9

encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
```



- following is the flow of encoder_self_attention_mask to incapsulate the connection of all the functions in transformer_model.py
- Note:  Each EncoderLayer is a module that contains operations like multi-head self-attention, feed-forward neural network, layer normalization, etc. When you call the forward method of the SequentialEncoder, it iterates through each EncoderLayer and applies them sequentially to the input tensors.



```bash
                     +------------------+
                     |   Transformer    |
                     +------------------+
                            |
                            |       +----------------+
                            +------>|     Encoder     |
                            |       +----------------+
                            |              |
                            |              |
                            |       +----------------+
                            +------>| Sequential     |
                            |       |   Encoder      |
                            |       +----------------+
                            |              |
                            |              |
                            |       +----------------+
                            +------>| EncoderLayer   |
                            |       +----------------+
                            |              |
                            |              |
                            |       +----------------+
                            +------>|  MultiHead     |
                            |       |  Attention     |
                            |       +----------------+
                            |              |
                            |              |
                            |       +----------------+
                            +------>|   Scaled Dot   |
                            |       |   Product      |
                            |       +----------------+
```



- ### Multi-Head Self Attention


<div align="center">
  <img src="https://deepfrench.gitlab.io/deep-learning-project/resources/transformer.png" alt="Image Alt Text" width='600'>
</div>

- Apply self-attention multiple times in parallel. For each head(self-attention layer) we use different W_q, W_k, W_v then concatenate the results —> A.
- the weights for q, k, and v are learned through the linear transformation performed by the self.qkv_layer (nn.Linear) in the MultiHeadAttention class. The linear layer learns the weights that transform the input x into query, key, and value representations for each head.
- num_head = 8 , head dimension is given by d_model // num_head. i.e 64
- self.num_head * self.head_dim = d_model (512)
-  output is in shape (batch_size, sequence_length, d_model)

- ### Multi-Head Cross Attention



<div align="center">
  <img src="https://vaclavkosar.com/images/cross-attention-in-transformer-decoder.png" alt="Image Alt Text" width= "400">
</div>


```bash
# here x is eng_batch and y is hin_batch 
                                      
 kv = self.kv_layer(x)
 q = self.q_layer(y)

```


- Except for inputs, cross-attention calculation is the same as Multi-Head Self Attention. Cross-attention combines asymmetrically two separate embedding sequences ( English from Encoder and Hindi from Decoder ) of same dimension, in contrast self-attention input is a single embedding sequence. One of the sequences serves as a query input, while the other as a key and value inputs.

<div align="center">
  <img src="https://vaclavkosar.com/images/cross-attention-in-transformer-architecture.png" alt="Image Alt Text" width = "400">
</div>


- Cross-attention is a mechanism that allows the decoder layers to incorporate information from the input sequence. This integration of information enables the decoder to predict the next token in the output sequence accurately.

- ### Addding Residual and Layer Normalization 


- Layer normalization is used in the Transformer model to help stabilize training, improve convergence, and mitigate the impact of input sequence length on the model's performance.
- There are many types of normalization where the axis varies along which normalization is carried out, it is shown as follows. Note that in this project we will be using Layer Normalization.

<div align="center">
  <img src="https://www.imaios.com/i/var/site/storage/images/0/8/3/4/464380-3-eng-GB/standardization_types.png?caption=1&ixlib=php-3.3.1&q=75&w=680" alt="Image Alt Text" width = "600">
</div>

- The Transformer model is a deep neural network architecture with multiple layers of self-attention and feed-forward neural networks. Each layer involves a series of matrix multiplications and non-linear transformations. During training, as gradients are propagated through these layers, the gradients can become very large or very small, leading to unstable training. This is known as the "exploding" or "vanishing" gradient problem.
- Layer normalization helps address these issues by normalizing the activations within each layer independently. It works similarly to batch normalization but operates along the feature dimension rather than the batch dimension. In layer normalization, the mean and variance of the activations are computed across the feature dimension (here it will be along d_model) for each individual example in the batch. The activations are then normalized based on these statistics, and learnable scale and shift parameters are applied to allow the model to learn the optimal scaling and translation


<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:796/1*_tXHN8LE-1LqrclrP9bCMg.png" alt="Image Alt Text" width ='250'>
</div>

```bash
class LayerNormalization(nn.Module):
    def __init__(self, parameter_shape, epsilon = 1e-5):
        super().__init__()
        self.parameter_shape = parameter_shape
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))

    def forward(self, input): # input --> (batch_size, sequence_len, d_model)
        dims = [-(i +1) for i in range(len(self.parameter_shape))] # len describes the dimention which should be used to compute mean.
        # here len is 1 --> renge is 0 --> so dims = [-1]--> so mean is computed along d_model 
        mean = input.mean(dim = dims, keepdim = True)  # normalised across d_model or embeddin_size (every sentence)
        var = ((input - mean)**2).mean(dim = dims, keepdim = True)
        std = (var + self.epsilon).sqrt()
        y = (input - mean)/std
        output = self.gamma * y + self.beta
        return output
```


```bash

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
		.
		.
        self.norm1 = LayerNormalization(parameter_shape = [d_model])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
		.
		.
		.
    def forward(self, x, self_attention_mask): 
        resudual_x = x.clone()
		.
		.
		.
        x = self.norm1(resudual_x + x)
        residual_x = x.clone()
		.
		.
		.
        x = self.norm2(residual_x + x)
        return x 


```
