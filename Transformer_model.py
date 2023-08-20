import numpy as np 
import torch 
from torch import nn 
import math  


#def get_device():
#    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v , mask = None): 
    d_k = q.size()[-1]  # (batch_size , num_head, sequemce_length, head_dim)
    scaled = torch.matmul(q, k.transpose(-1,-2))/ math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1,0,2,3) + mask # scaled permute to shape --> (num_head, batch_size, seq_len, seq_len) <-- also the shape of mask 
        scaled = scaled.permute(1,0,2,3) # permute is done twice now so its back to (batch_size, num_head, seq_len, seq_len)
    attention = torch.nn.functional.softmax(scaled, dim =-1)  # attention --> (batch_size, num_heads, seq_len_q, seq_len) softmax along seq_len
    values = torch.matmul(attention, v ) # values --> (batch_size, num_heads, seq_len_q, d_v)
    return attention, values
#value: Weighted sum of value vectors based on attention scores. Shape: (batch_size, num_heads, seq_len_q, d_v)
#attention: Attention scores obtained from softmax of scaled dot-product scores. Shape: (batch_size, num_heads, seq_len_q, seq_len_k)

#Initially, the shape of scaled after matrix multiplication q and k is (batch_size, num_heads, seq_len_q, seq_len_k).
#After adding the mask and the first permutation, the shape becomes (num_heads, batch_size, seq_len_q, seq_len_k).
#Then, after the second permutation, the shape becomes (batch_size, num_heads, seq_len_q, seq_len_k) again

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

    def forward(self):
        #calculations are performed element-wise across the entire tensor using PyTorch's broadcasting and element-wise operations.
        even_i = torch.arange(0, self.d_model, 2 ).float()   #(start: Number, end: Number, step: Number)
        denominator = torch.pow(10000, even_i/self.d_model)  #(input: Tensor, exponent: Tensor)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length,1)) #--> 1d to 2d ( column vector )
        even = torch.sin(position/ denominator)
        odd = torch.cos(position/ denominator)
        stacked = torch.stack([even, odd], dim = 2)
        pos_enc = torch.flatten(stacked, start_dim=1, end_dim=2)
        return pos_enc
    
class SentenceEmbedding(nn.Module):
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        #nn.Embedding is a class provided by PyTorch that represents an embedding layer for categorical data, such as words or tokens.
        self.embedding = nn.Embedding(self.vocab_size, d_model) # english->(69,512), hindi->(100,512)
        self.language_to_index = language_to_index
        self.positional_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1) #if p=0.1 means that during training, each element in the input_tensor has a 10% chance of being set to zero (dropped out).
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token, end_token):
        def tokenizer(sentence, start_token , end_token):
            sentence_word_indices = [self.language_to_index[token] for token in list(sentence) ] # list Convert the sentence to a list of characters
            if start_token:
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN]) # begins with START_TOKEN
            if end_token:
                sentence_word_indices.append(self.language_to_index[self.END_TOKEN]) # ends with END_TOKEN
            for _ in range(len(sentence_word_indices), self.max_sequence_length):
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN]) # padding the remaining length ( max_sequence_length - len(sentence) ) with PADDING_TOKEN
            return torch.tensor(sentence_word_indices)
        
        tokenized = []
        for sentence_num in range(len(batch)):  
            tokenized.append(tokenizer(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized#.to(get_device())
    
    def forward(self, x , start_token, end_token):
        x = self.batch_tokenize(x , start_token, end_token)
        x = self.embedding(x)
        pos = self.positional_encoder()#.to(get_device())
        x = self.dropout(x + pos)
        return x
    
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model//num_head
        self.qkv_layer = nn.Linear(d_model, 3*d_model) # querey, key and value 
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size() # x --> tuple(batch_size, sequence_length, d_model)
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_head, 3*self.head_dim) # 3xd_model = 3xhead_dim * num_head
        qkv = qkv.permute(0,2,1,3)
        q, k, v = qkv.chunk(3, dim=-1) # split into 3 
        #q, k, v = qkv.chunk(3, dim=1): This line uses the chunk function to split the qkv tensor into three tensors 
        # along the second dimension (which now corresponds to the sequence length). Each resulting tensor will have 
        # the shape (batch_size, num_heads, sequence_length, head_dim).
        attention, value = scaled_dot_product(q,k,v,mask) # we mask values --> (batch_size, num_heads, seq_len_q, d_v)
        value = value.permute(0,2,1,3).reshape(batch_size,sequence_length, self.num_head * self.head_dim) # self.num_head * self.head_dim = d_model (512)
        output = self.linear_layer(value) # input and output shapes --> (batch_size, sequence_length, d_model)
        return output    
#The shape of value will be (batch_size, num_heads, seq_len_q, d_v). Here, d_v represents the dimension of the value vectors, which is typically smaller than d_model.
#The shape of output will be (batch_size, sequence_length, d_model).
#Yes, q, k, and v are separated at the end using the chunk operation.
#The output of the forward function is called output, which contains the final representations after the multi-head attention mechanism and linear transformation

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
        return output #--> output shape same as input shape: (batch_size, sequence_len, d_model)
    
#ReLU activation is applied along the d_model dimension
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x) # linear transformation to shape (batch_size, max_sequence_length, hidden)
        x = self.relu(x) # f(x) = max(0, input) --> -ve becomes 0, +ve remains the same 
        x = self.dropout(x)
        x = self.linear2(x) # linear transformation to shape (batch_size, max_sequence_length, d_model)
        return x # x shape returned is (batch_size, max_sequence_length, d_model)
    
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
    
# the asterisk (*) before a parameter in a function definition is used to collect any number of positional arguments into a tuple
# It uses the *inputs syntax to indicate that you can pass additional positional arguments to the method,
#  but these arguments will be ignored since the method specifically expects only two arguments.
class SequentialEncoder(nn.Sequential):
    def forward(self, *input):
        x, self_attention_mask = input
        for module in self._modules.values():
            x = module(x, self_attention_mask) # passes the value of x and self_attention mask to every layer in EncoderLayer for every instance 
        return x #The SequentialEncoder takes care of applying each EncoderLayer sequentially and returning the final output.
    
class Encoder(nn.Module):
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
        self.sentence_embedding = SentenceEmbedding( max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]) # * each element is an instance of EncoderLayer in a list comprehension  
        #However, in this context, the values of x and self_attention_mask are not assigned yet.
        #They will be assigned when you actually call the forward method of the SequentialEncoder instance (self.layers) with appropriate arguments.
        #So, before the forward method is called, the values of x and self_attention_mask within each EncoderLayer instance are still uninitialized (None)
    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x ,start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x 
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model//num_head
        self.kv_layer = nn.Linear(d_model, 2*d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask): # x is the output of encoder , y is the output of decoder 
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_head, 2*self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_head, self.head_dim)
        kv = kv.permute(0,2,1,3)
        q = q.permute(0,2,1,3)
        k, v = kv.chunk(2, dim = -1)
        attention, value = scaled_dot_product(q,k, v, mask)
        value = value.permute(0,2,1,3).reshape(batch_size, sequence_length, d_model)
        output = self.linear_layer(value)
        return output
    

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
    
class SequentialDecoder(nn.Sequential):
    def forward(self, *input):
        x, y , self_attention_mask , cross_attention_mask = input
        for module in self._modules.values():
            y = module(x, y , self_attention_mask, cross_attention_mask)
        return y 
    
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

    def forward(self, 
                x,
                y,
                encoder_self_attention_mask = None,
                decoder_self_attention_mask = None,
                decoder_cross_attention_mask = None,
                encoder_start_token = False,
                encoder_end_token = False,
                decoder_start_token = True,
                decoder_end_token = False):
        x = self.encoder(x, encoder_self_attention_mask, encoder_start_token, encoder_end_token)
        output = self.decoder(x, y , decoder_self_attention_mask, decoder_cross_attention_mask, decoder_start_token, decoder_end_token)
        output = self.linear(output)
        return output
# x --> eng_batch, y --> hin_batch
# Example of using Encoder <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
''' Parameters
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 3
max_sequence_length = 100
language_to_index = {'word1': 0, 'word2': 1, ...}
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PADDING_TOKEN = '<PAD>'

# Create an instance of Encoder

encoder_instance = Encoder(  <-- here 
    d_model=d_model,
    ffn_hidden=ffn_hidden,
    num_heads=num_heads,
    drop_prob=drop_prob,
    num_layers=num_layers,
    max_sequence_length=max_sequence_length,
    language_to_index=language_to_index,
    START_TOKEN=START_TOKEN,
    END_TOKEN=END_TOKEN,
    PADDING_TOKEN=PADDING_TOKEN
)

# Example input data
batch_size = 32
sequence_length = 50
x = torch.randn(batch_size, sequence_length, d_model)  # Example input data
self_attention_mask = torch.ones(num_heads, batch_size, sequence_length, sequence_length)  # Example attention mask
start_token = language_to_index[START_TOKEN]
end_token = language_to_index[END_TOKEN]

# Call the forward method of encoder_instance
output = encoder_instance(x, self_attention_mask, start_token, end_token) <-- here 
'''






#x = get_device()
#print(x) 










    

        


























    

