# English to Hindi Transformer Translation

This repository contains code explanation and concepts to know for translating English text to Hindi using a Transformer architecture with attention mechanisms.

<div align="center">
  <img src="https://i0.wp.com/www.michigandaily.com/wp-content/uploads/2021/03/0-1.jpg?fit=1200%2C960&ssl=1" alt="Image Alt Text" width='700'>
</div>

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Transformer Architecture](#transformer-architecture)
- [Results](#results)
- [Future Scope](#future-scope)

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

- as gradients are propagated backward during training, they can become very small due to repeated multiplication by small weights. This leads to gradients that vanish, causing slow or even stalled learning in deeper layers of the network. To deal with this problem residue is added before normalization.

```bash

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
		.
		.
        self.norm1 = LayerNormalization(parameter_shape = [d_model])
        self.norm2 = LayerNormalization(parameter_shape = [d_model])



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

- ### Feed Forward Neural Network

- In this project we have used 1048 nodes in the hidden layer. 
- there are 2 hidden layers. First a linear transformation is applied to give output shape (batch_size, max_sequence_length, hidden) from (batch_size, max_sequence_length, d_model)
- ReLU Activation function is applied over this output i.e f(x) = max(0, input) --> -ve becomes 0, +ve remains the same 
- Followed by a Dropout layer with drop_out probability of 10%.
- Finally the output is transformed linearly back to its original shape —> (batch_size, max_sequence_length, d_model)
- Feed Forward Layer is used in both Encoder as well as Decoder structure in Transformer.

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:791/0*hzIQ5Fs-g8iBpVWq.jpg" alt="Image Alt Text" width ='500'>
</div>

- ### Why use a Feed-Forward NN?

- It is a Multi layer perceptron, its main purpose is to transform the attention vectors into a form that is acceptable by the next encoder or decoder layer.
-  During training, the weights of the linear layers in the FFN are updated through backpropagation, just like in any other neural network. The model learns to adjust these weights to minimize the loss function associated with the task being performed, such as language translation.
- ### Which Weights are Updated: 
- The weights that are updated during training are the weights of the linear layers within the FFN. The weights of other components in the model, such as the attention mechanisms, positional encodings, and embeddings, are updated as well. The entire model is trained in an end-to-end manner to jointly optimize all components for the specific task.
In summary, the feed-forward neural network in the Transformer model enhances the model's ability to capture complex patterns and relationships in the data by introducing non-linearity and position-wise transformations. The weights of the FFN's linear layers are updated during training through backpropagation, along with other weights in the model, to learn representations that improve the model's performance on the given task.



<ul>
  <li>
    <h3>Encoder</h3>
    <ul>
      <li>
        The EncoderLayer comprises of:
        <ul>
          <li>Multi-Head Self Attention Layer</li>
          <li>Layer Normalization and adding residue</li>
          <li>Feed Forward NN</li>
          <li>Layer Normalization and adding residue</li>
          <li>Passing the output (K and V) along Decoder</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>



<div align="center">
  <img src="https://wikidocs.net/images/page/162096/3_encoder_decoder_layer_class.png" alt="Image Alt Text" width = '600'>
</div>


- In the above image the num of layers is 6 ( according to “Attention is All you Need”). However in this project I have used just 1 layer becuase of computational expense.

```bash 
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model = d_model, num_head = num_heads)
        self.norm1 = LayerNormalization(parameter_shape = [d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm2 = LayerNormalization(parameter_shape = [d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask): # x --> (batch_size, max_sequence_len, d_model), self_attention_maks --> (num_head, batch_size, max_sequence_len, max_sequence_len)
        resudual_x = x.clone()
        x = self.attention(x, self_attention_mask) # (batch_size, sequence_length, d_model)
        x = self.dropout1(x)
        x = self.norm1(resudual_x + x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(residual_x + x)
        return x 
    
```




```bash
class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,  # <----- num_layers = 1
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding( max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
      
    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x ,start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x 

```
- x is English batch 
- Important thing to note here is in Encoder language_to_index is English vocabulary mapping and in Decoder it is Hindi vacabulary mapping.

<ul>
  <li>
    <h3>Decoder</h3>
    <ul>
      <li>
        The DecoderLayer comprises of:
        <ul>
          <li>Multi-Head Self Attention Layer</li>
          <li>Layer Normalization and adding residue</li>
          <li>Multi-Head Cross Attention Layer</li>
          <li>Layer Normalization and adding residue</li>
          <li>Feed Forward NN</li>
          <li>Layer Normalization and adding residue</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>


```bash
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=num_heads)
        self.layer_norm1 = LayerNormalization(parameter_shape=[d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_head=num_heads)
        self.layer_norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameter_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y , self_attention_mask, cross_attention_mask):
        residual_y = y.clone()
        y = self.self_attention(y, self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + residual_y)

        residual_y = y.clone()
        y = self.encoder_decoder_attention(x, y, cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(residual_y + y )

        residual_y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(residual_y + y)
        return y 

```
- y is hindi batch 

- Decoder is also run just once I.e here num_layers = 1, However can be changed while hyper parameter tuning for better results. 
```bash
class Decoder(nn.Module):
    def __init__(self,
                 d_model, 
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layer = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x , y , self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layer(x, y , self_attention_mask, cross_attention_mask)
        return y 
```

<h2> Transformer</h2>
<p> Following is the execution of Encoder and Decoder in Transformer class. </p>

```bash
class Transformer(nn.Module):
    def __init__(self, 
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 d_output,
                 english_to_index,
                 hindi_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, hindi_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, d_output)
```
<p> here d_output is hindi vocabulary as we need output in Hindi language.</p>
<p> To achieve that we use a linear transformation layer at the end 
	
```bash
 self.linear = nn.Linear(d_model, d_output)
```




<h2>Results</h2>

```bash
translation = translate("i hope this works.")
print(translation)
```

```bash
मुझे लगता है कि यह काम कर रहा है।<END>
```

```bash
translation = translate("what should we do when the day starts?")
print(translation)
```

```bash
क्या है कि क्या करने के लिए क्या है?<END>
```

<p> This results is generated after training for 6 epochs, where each epoch has 5194 iterations  </p>

```bash
batch_size = 30
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)
```

```bash
len(iterator) # --> 5194
```

<p> With more training and increasing the num of layers in Encoder and Decoder model we can achieve more accurate results. </p>


 
<h2>Future Scope</h2>
<p>In order to execute this code, I used Google Colab Pro+ as it is computationally expensive to train this model. However, I used just 1 layer for the encoder and decoder and employed 6 epochs. I used around 250 compute units. If you want to check out the price ranges for Google Colab, you can find more information <a href="https://colab.research.google.com/signup">here</a>.</p>
<p>Adding an accuracy metric like Translation Edit Rate (ETR). I am working on it and will upload results soon. </p>


