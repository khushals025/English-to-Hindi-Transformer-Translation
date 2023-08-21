# English to Hindi Transformer Translation

This repository contains code for translating English text to Hindi using a Transformer architecture with attention mechanisms.

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

### Positional Encoding


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

### How to Encode word positions?


We use sin and cosine functions of different frequencies.We represent each token as a vector of 512 dimensions (from element 0 to element 511). A position (pos) is an integer from 0 to a pre-defined maximum number of tokens in a sentence (a.k.a. max length) minus 1. For example, if the max length is 128, positions are 0 to 127.
Since we have 512 dimensions, we have 256 pairs of sine and cosine values. As such, the value of i goes from 0 to 255.

<div align="center">
  <img src="https://i.stack.imgur.com/67ADh.png" alt="Image Alt Text" width="400">
</div>

Here, d_model = 512
pos is integer within range 0 <= pos < T, where T is input sequence size or max_sequence_len


### Sentence Embedding

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
- input data
- Embeddings
- Self Attention Mechanism
- Cross Attention Mechanism
- Masking
- Feed Forward NN
- Add and Normalization
 


