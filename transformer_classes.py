import torch
import torch.nn as nn
import numpy as np

"""
following Umar Jamil's  Transformer from Scratch tutorial
https://www.youtube.com/watch?v=ISNdQcPhsts

implements NN from Attention is All You Need paper

"""

##### INPUTS TO THE MULTI-HEAD ATTENTION #####

class InputEmbeddings(nn.Module):
        
    def __init__(self, d_model:int, vocab_size:int): 

        """
        Converts the input sentence into embeddings.

        Args:
        d_model : dimension of the embedding (for each word)
        vocab_size : size of the vocabulary corpus
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        #nn.Embedding is a lookup table that stores embeddings of a fixed dictionary and size 
    
    def forward(self, x):
        #(batch, seq_len) --> (batch, seq_len, d_model)
        x = self.embedding(x)*np.sqrt(self.d_model)
        return x
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout_rate = None):

        """
        Creates the positional encodings for the 'words' in the input sentence.
        -- technically it's the embeddings, not the words

        Args:
        d_model : dimension of the embedding (for each word)
        seq_len : length of the input sentence
        dropout_rate : proportion of units that are dropped
        """

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = None
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate) #why tho

        #prepare tensor of shape (seq_len, d_model)
        pos_enc = torch.zeros(size=(seq_len,d_model))

        #tensor of shape (seq_len, 1) for the position dimension
        position = torch.arange(0, seq_len, dtype= torch.float).unsqueeze(1) #unsqueeze adds empty dimension
        #prepare the divisor term (see paper), but with log
        #i.e we have i * log(10000)/d_model
        #giving a tensor of length d_model/2
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        #create the encodings
        pos_enc[:,0::2] = torch.sin(position*div_term) #for the even positions (0,2,4,6,...)
        pos_enc[:,1::2] = torch.cos(position*div_term) #for the even positions (0,1,3,5,...)

        pos_enc = pos_enc.unsqeeze(0) #gives shape (1,seq_len, d_model)

        self.register_buffer('pos_enc', pos_enc) #save this during training, but keep fixed


    def forward(self,x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        x = x + (self.pos_enc[:,:x.shape[1],:]).requires_grad_(False) #ensure not updated during training
        if self.dropout is not None:
            x = self.dropout(x)
        return x



##### ADDITIONAL LAYERS #####
    
class LayerNormalization(nn.Module):
    #compare to original torch implementation (https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
    #what should be the dimension of gamma and bias?
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon #to avoid divide by 0
        self.gamma = nn.Parameter(torch.ones(1)) #learnable weight
        self.beta = nn.Parameter(torch.zeros(1)) #learnable bias
    
    def forward(self,x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        mean = torch.mean(x, dim = -1, keepdim = True) #keepdim for broadcasting
        st_dev = torch.std(x, dim = -1, keepdim = True)
        layer_output = self.gamma*(x-mean)/(st_dev + self.epsilon) + self.beta
        #the gamma and beta act as learnable weight and bias (see torch LayerNorm)
        #so the output is not simply a distribution with mean 0 & st dev of 1
        return layer_output


class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int, dropout_rate: float, d_ff:int = 2048):
        """
        Feed-forward block made of 2 linear layers

        Args:
        d_model : dimension of the embedding
        d_ff : inner layer dimensionality (i.e. number of units in first layer)
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

#class residual_connection(nn.Module):
# decided not to make a separate class for this, add directly in encoder


##### ATTENTION BLOCK #####

class SingleAttention(nn.Module):

    def __init__(self, dropout_layer:nn.Dropout = None):
        super().__init__()
        self.dropout = dropout_layer
    
    def forward(self, query, key, value, mask=None):
        # query, key, and value should all have shape (batch, seq_len, d_k)
        d_k = query.shape[-1]
        query_key = torch.matmul(query, key.transpose(-2,-1)) / np.sqrt(d_k)
        # (batch,..., seq_len, d_k) --> (batch, ..., seq_len, seq_len)

        if mask is not None:
            query_key.masked_fill_(mask == 0, value = -1e9)
            #if the mask value is 0, fill with "negative infinity", so that the softmax is zero
        
        if self.dropout is not None:
            query_key = self.dropout(query_key)
        
        query_key = nn.Softmax(dim=-1)(query_key) #so that the ROWS add up to 1
        #the video seem to be missing this softmax?
        attention_output = torch.matmul(query_key, value)
        # (batch,..., seq_len, seq_len) --> (batch, ..., seq_len, d_model)

        return attention_output, query_key


class multi_head_attention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "embedding dimension d_model is not divisible by number of heads n_heads"
        self.d_k = int(d_model/n_heads)

        #make linear layers to the model can learn WQ, WK, and WV 
        #see page 5 of paper
        #decided to follow the video in making d_model --> d_model and then split, instead of making d_model --> d_k
        #should be equivalent?
        self.wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)

        self.wo = nn.Linear(d_model,d_model)

        self.attention = SingleAttention()

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p= dropout_rate)
            self.attention = SingleAttention(dropout_layer=self.dropout)

    def forward(self,q,k,v,mask):
        query = self.wq(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.wk(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.wv(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # note: explanation of torch.Tensor.view:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

        attention_output, query_key = self.attention(query, key, value, mask)

        # Concatenate the heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = attention_output.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        #note: explanation of torch.Tensor.contiguous:
        #https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch 

        x = self.wo(x) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return x








