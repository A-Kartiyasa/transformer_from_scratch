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

        pos_enc = pos_enc.unsqueeze(0) #gives shape (1,seq_len, d_model)

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
    def __init__(self, features:int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon #to avoid divide by 0
        self.gamma = nn.Parameter(torch.ones(features)) #learnable weight, per feature (which is usually d_model)
        self.beta = nn.Parameter(torch.zeros(features)) #learnable bias
    
    def forward(self,x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        mean = torch.mean(x, dim = -1, keepdim = True) #keepdim for broadcasting
        st_dev = torch.std(x, dim = -1, keepdim = True)
        layer_output = self.gamma*(x-mean)/(st_dev + self.epsilon) + self.beta
        #the gamma and beta act as learnable weight and bias (see torch LayerNorm)
        #so the output is not simply a distribution with mean 0 & st dev of 1
        return layer_output


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, dropout_rate: float, d_ff:int = 2048):
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

class OutputLayer(nn.Module): #returns a probability of each word in the vocabulary

    def __init__(self,d_model:int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self,x):
        #(batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        x = self.linear(x)
        x = torch.log_softmax(x, dim=-1)
        return x


##### ATTENTION LAYERS #####

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


class MultiHeadAttention(nn.Module):

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


    def forward(self,q,k,v,mask=None):
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
        #apparently torch.Tensor.view needs a contiguous input.

        x = self.wo(x) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return x, query_key


##### ENCODER AND DECODER BLOCKS #####

class EncoderBlock(nn.Module):

    def __init__(self, features:int,
                 self_attention_block: MultiHeadAttention, 
                 feed_forward_block: FeedForwardBlock,
                 dropout_rate: float = None):
        #self-attention & feedforward as argumentshere in case want to add other custom layers

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.norm = LayerNormalization(features)
        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None
    
    def forward(self, x, src_mask):
        # x should be input embedding + positional encoding
        #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        attention_output = self.self_attention_block(x,x,x,src_mask)
        if self.dropout is not None:
            attention_output = self.dropout(attention_output)
        x = self.norm(x+attention_output) #with residual connection
        feed_forward_ouput = self.feed_forward_block(x)
        if self.dropout is not None:
            feed_forward_ouput = self.dropout(feed_forward_ouput)
        x = self.norm(x+feed_forward_ouput)
        
        return x


class DecoderBlock(nn.Module):

    def __init__(self, features:int, 
                 self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock,
                 dropout_rate: float = None):

        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.norm = LayerNormalization(features)
        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        ### self attention
        self_attention_output = self.self_attention_block(x,x,x,tgt_mask)
        if self.dropout is not None:
            self_attention_output = self.dropout(self_attention_output)
        x = self.norm(x+self_attention_output) #with residual connection

        ### cross attention
        # query is output of previous decoder layer, key & value is final output of encoder
        cross_attention_output = self.cross_attention_block(x,encoder_output, encoder_output, src_mask)
        if self.dropout is not None:
            cross_attention_output = self.dropout(cross_attention_output)
        x = self.norm(x+cross_attention_output)

        feed_forward_ouput = self.feed_forward_block(x)
        if self.dropout is not None:
            feed_forward_ouput = self.dropout(feed_forward_ouput)
        x = self.norm(x+feed_forward_ouput)
        
        return x


##### PLACEHOLDER CLASSES TO CHAIN MULTIPLE ENCODER (DECODER) BLOCKS ######

class Encoder(nn.Module): 

    def __init__(self, features:int, layers: nn.ModuleList): 
        #layers should be a list of encoder blocks, see build_transformer below
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, features:int, layers: nn.ModuleList):
        #layers should be a list of decoder blocks, see build_transformer below
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self,x,encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

##### TRANSFORMER NETWORK #####

class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder: Decoder,
                 src_embed = InputEmbeddings, tgt_embed = InputEmbeddings,
                 src_pos = PositionalEncoding, tgt_pos = PositionalEncoding,
                 output_layer = OutputLayer):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.output_layer = output_layer
    
    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      n_blocks: int=6, n_heads: int=8,  
                      d_ff: int=2048, dropout_rate: float= None):
    
    ### Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    ### Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout_rate)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout_rate)

    ### Create the encoder blocks
    encoder_blocks = []
    for _ in range(n_blocks):
        encoder_self_attention_block = MultiHeadAttention(d_model, n_heads, dropout_rate)
        feed_forward_block = FeedForwardBlock(d_model, dropout_rate, d_ff)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, 
                                     feed_forward_block, dropout_rate)
        encoder_blocks.append(encoder_block)

    ### Create the decoder blocks
    decoder_blocks = []
    for _ in range(n_blocks):
        decoder_self_attention_block = MultiHeadAttention(d_model, n_heads, dropout_rate)
        decoder_cross_attention_block = MultiHeadAttention(d_model, n_heads, dropout_rate)
        feed_forward_block = FeedForwardBlock(d_model, dropout_rate, d_ff)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, 
                                     decoder_cross_attention_block, feed_forward_block, 
                                     dropout_rate)
        decoder_blocks.append(decoder_block)

    ### Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    ### Create the projection layer
    projection_layer = OutputLayer(d_model, tgt_vocab_size)
    
    ### Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    ### Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer