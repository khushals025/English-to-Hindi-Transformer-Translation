# English to Hindi Transformer Translation

This repository contains code for translating English text to Hindi using a Transformer architecture with attention mechanisms.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
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
I have used English to Hindi Translation datset in this repository.
- dataset: <https://ai4bharat.iitm.ac.in/samanantar>

## Transformer Architecture
- input data
- Embeddings
- Self Attention Mechanism
- Cross Attention Mechanism
- Masking
- Feed Forward NN
- Add and Normalization
 


