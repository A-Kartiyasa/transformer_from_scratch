import torch
import torch.nn as nn
import numpy as np

"""
following Umar Jamil's  Transformer from Scratch tutorial
https://www.youtube.com/watch?v=ISNdQcPhsts

implements NN from Attention is All You Need paper

"""

##################### INPUTS TO THE MULTI-HEAD ATTENTION #####################

class InputEmbeddings(nn.Module):
        
    def __init__(self, d_model:int, vocab_size:int): 

        """
        Converts the input sentence into embeddings.

        Init Args:
            d_model     : dimension of the embedding (for each word)
            vocab_size  : size of the vocabulary corpus
            
        """
        
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        #nn.Embedding is a lookup table that stores embeddings of a fixed dictionary and size 
    
    def forward(self, x):
        
        """ 
        Args:
            x : input sentence(s) of shape (batch, seq_len)
        
        Returns:
            x   : Embedded sentence(s) of shape (batch, seq_len, d_model)     
        """
        #(batch, seq_len) --> (batch, seq_len, d_model)
        x = self.embedding(x)*np.sqrt(self.d_model) #see section 3.4 in the paper
        return x
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout_rate = None):

        """
        Creates sine and cosine positional encodings for the 'words' in the input sentence,
        -- technically it's the embeddings, not the words
        and adds the positional encoding into the input sentence.
        See section 3.5 of the paper.

        Init Args:
            d_model         : dimension of the embedding (for each word)
            seq_len         : length of the input sentence
            dropout_rate    : proportion of units that are dropped
            
        """

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = None
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate) 

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
        
        """
        Args:
            x   : input sentence(s) of shape (batch, seq_len, d_model).
            
        Returns:
            x   : input sentence(s) with positional encoding added, shape (batch, seq_len, d_model).
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        x = x + (self.pos_enc[:,:x.shape[1],:]).requires_grad_(False) 
        #x.shape[1] is seq_len
        #requires_grad_(False) ensure not updated during training
        x = self.dropout(x) #see section 5.4 of paper, apply dropout to sum of embedding and pos enc
        return x



##################### ADDITIONAL LAYERS #####################
    
class LayerNormalization(nn.Module):
    
    #compare to original torch implementation (https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
    #what should be the dimension of gamma and bias?
    
    def __init__(self, features:int, epsilon: float = 1e-5):
        
        """
        Normalizes the input across all units in the layer, for a single example.
    
        Init Args:
            features    : number of units in the layer. Usually should be d_model.
            epsilon     : a small number added to the divisor (i.e. standard deviation) to avoid division by 0.
        
        """
         
        super().__init__()
        self.epsilon = epsilon #to avoid divide by 0
        self.gamma = nn.Parameter(torch.ones(features)) #learnable weight, per feature (which is usually d_model)
        self.beta = nn.Parameter(torch.zeros(features)) #learnable bias
    
    
    def forward(self,x):
        
        """ 
        Args:
            x : input of shape (batch, seq_len, d_model).
            
         Returns:
            layer_output : normalized output of shape (batch, seq_len, features). 
        
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        mean = torch.mean(x, dim = -1, keepdim = True) #keepdim for broadcasting
        st_dev = torch.std(x, dim = -1, keepdim = True)
        layer_output = self.gamma*(x-mean)/(st_dev + self.epsilon) + self.beta
        #the gamma and beta act as learnable weight and bias (see torch LayerNorm)
        #so the output is not simply a distribution with mean 0 & st dev of 1
        return layer_output


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int = 2048):
        
        """
        Feed-forward block made of 2 linear layers.
        See section 3.3 in the paper.

        Init Args:
            d_model : dimension of the embedding
            d_ff    : inner layer dimensionality (i.e. number of units in first layer)
        
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        
        """ 
        Args:
            x: input of shape (batch, seq_len, d_model).
        
        Returns:
            x: output of shape (batch, seq_len, d_model)
            
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        x = torch.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

#class residual_connection(nn.Module):
# decided not to make a separate class for this, add directly in encoder

class OutputLayer(nn.Module): 

    def __init__(self,d_model:int, vocab_size:int):
        
        """ 
        Fully-connected layer to project the decoder output into vocab size.
        Takes d_model inputs, and has vocab_size units.
    
        Init Args:
            d_model     : size of the embeddings used in the model
            vocab_size  : size of the vocabulary used to train the model
        """
        
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self,x):
        
        """ 
        Args:
            x: input of shape (batch, seq_len, d_model).
        
        Returns:
            x: output of shape (batch, seq_len, vocab_size).
        """
        #(batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        x = self.linear(x)
        #x = torch.log_softmax(x, dim=-1) #skip this, we will be using cross entropy loss later
        return x


#####################  ATTENTION LAYERS ##################### 

class SingleAttention(nn.Module):

    def __init__(self):
        
        """ 
        Implements scaled dot-product attention.
        See section 3.2.1 in the paper.
        
        """
        super().__init__()
    
    def forward(self, query, key, value, mask=None):
        
        """ 
        Args: 
            query   : input "sentence" of shape (batch, ..., seq_len, d_k)
            key     : input "sentence" of shape (batch, ..., seq_len, d_k)
            value   : input "sentence" of shape (batch, ..., seq_len, d_k)
            mask    : mask, used for example in the decoder block to prevent the model from 'seeing' subsequent words
                        in the input sentence when predicting the next word.
        
        Returns:
            attention_output    : matmul(query_key,V), of shape (batch, ..., seq_len, d_k)
            query_key           : the "attention score", softmax(QK/sqrt(d_k)), of shape (batch,..., 1)
            
        """
        
        # query, key, and value should all have shape (batch, seq_len, d_k)
        d_k = query.shape[-1]
        query_key = torch.matmul(query, key.transpose(-2,-1)) / np.sqrt(d_k)
        # (batch,..., seq_len, d_k) --> (batch, ..., seq_len, seq_len)

        if mask is not None:
            query_key.masked_fill_(mask == 0, value = -1e9)
            #if the mask value is 0, fill with "negative infinity", so that the softmax is zero
        
        query_key = nn.Softmax(dim=-1)(query_key) #so that the ROWS add up to 1
        #the video seem to be missing this softmax?
        attention_output = torch.matmul(query_key, value)
        # (batch,..., seq_len, seq_len) --> (batch, ..., seq_len, d_k)

        return attention_output, query_key


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):

        """ 
        Implements multi-head attention with learnable weights.
        See section 3.2.2 in the paper.
        
        The learnable weights (wq, wk, wv, wo) are implemented as linear layer with input units = output units = d_model.
        wq - query weight, wk - key weight, wv - value weight, wo - overall weight.
        
        The query, key, and value tensors, which has shape (batch, seq_len, d_model), 
        are first multiplied by the corresponding weights
        and then split and rearranged into shape (batch, n_heads, seq_len, d_k), where d_k = d_model / n_heads.
        
        Scaled dot-product attention is then calculated for each head using the SingleAttention module.
        
        Init Args:
            d_model: size of the embedding used in the model
            n_heads : number of attention heads
        
        """

        super().__init__()
        self.d_model = d_model
        self.h = n_heads
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


    def forward(self,q,k,v,mask=None):
        
        """ 
        Args:
            q: query "sentence" of shape (batch, seq_len, d_model).
            k: key "sentence" of shape (batch, seq_len, d_model).
            v: value "sentence" of shape (batch, seq_len, d_model).
        
        Returns:
            x: output of shape (batch, seq_len, d_model).
        """
        
        query = self.wq(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.wk(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.wv(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # note: explanation of torch.Tensor.view:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

        attention_output, self.query_key = self.attention(query, key, value, mask)

        # Concatenate the heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = attention_output.transpose(1, 2)
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k) #separate this to avoid UnboundLocalError: cannot access local variable 'x' where it is not associated with a value
        #note: explanation of torch.Tensor.contiguous:
        #https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch 
        #apparently torch.Tensor.view needs a contiguous input.

        x = self.wo(x) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return x


##################### ENCODER AND DECODER BLOCKS #####################


class EncoderBlock(nn.Module):
    
    

    def __init__(self, features:int,
                 self_attention_block: MultiHeadAttention, 
                 feed_forward_block: FeedForwardBlock,
                 dropout_rate: float):
        #self-attention & feedforward as arguments here in case want to add other custom layers

        """ 
        Assembles a single encoder block. 
        See section 3.1 of the paper.
        
        The encoder block consists of 2 sub-layers:
            1. Self-attention layer. 
            Computes the self-attention output, adds the residual connection, then applies layer norm.
            The query, key, and value for the Attention computation is the same input "sentence".

            2. Feed-forward layer
            Computes the feed-forward output, adds the residual connection, then applies layer norm.
        
        Init Args:
            features            : usually should be dimensions of the embedding (d_model).
            self_attention_block: an instance of the MultiHeadAttention class.
            feed_forward_block  : an instance of the FeedForwardBlock class.
            dropout_rate        : dropout probability.
            
        """

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.norm = LayerNormalization(features)
        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None
    
    def forward(self, x, src_mask):
        
        """ 
        Args:
            x           : position-encoded input "sentence", of shape (batch, seq_len, d_model).
            src_mask    : mask to be applied to the Encoder input
        
        Returns:
            x: output of shape (batch, seq_len, d_model)
        """
        # x should be input embedding + positional encoding
        #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        attention_output = self.self_attention_block(x,x,x,src_mask)
        attention_output = self.dropout(attention_output) #see paper section 5.4, dropout after every sub-layer
        x = self.norm(x+attention_output) #with residual connection
        feed_forward_output = self.feed_forward_block(x)
        feed_forward_output = self.dropout(feed_forward_output) #see paper section 5.4, dropout after every sub-layer
        x = self.norm(x+feed_forward_output)
        
        return x


class DecoderBlock(nn.Module):

    def __init__(self, features:int, 
                 self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock,
                 dropout_rate: float):
        
        """
        Assembles a single Decoder block. 
        See section 3.1 of the paper.

        The Decoder block consists of 3 sub-layers:
            1. Self-attention layer
            Computes the self-attention output, adds the residual connection, then applies layer norm.
            The query, key, and value for the Attention computation is the same input "sentence".

            2. Cross-attention layer
            Computes the cross-attention output, adds the residual connection, then applies layer norm.
            The query is the output of the previous layer (self-attention), while the key & value is the output of the entire Encoder side (not the output of one Encoder block).

            3. Feed-forward layer
            Computes the feed-forward output, adds the residual connection, then applies layer norm.

        
        """

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

        """
        Args:
            x               : position-encoded input "sentence", of shape (batch, seq_len, d_model).
            encoder_output  : output of the entire Encoder
            src_mask        : mask to be applied to the Encoder input
            tgt_mask        : mask to be applied to the Decoder input

        """

        #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        ### self attention
        self_attention_output = self.self_attention_block(x,x,x,tgt_mask)
        self_attention_output = self.dropout(self_attention_output) #see paper section 5.4, dropout after every sub-layer
        x = self.norm(x+self_attention_output) #with residual connection

        ### cross attention
        # query is output of previous decoder layer, key & value is final output of encoder
        cross_attention_output = self.cross_attention_block(x,encoder_output, encoder_output, src_mask)
        cross_attention_output = self.dropout(cross_attention_output) #see paper section 5.4, dropout after every sub-layer
        x = self.norm(x+cross_attention_output)

        feed_forward_output = self.feed_forward_block(x)
        feed_forward_output = self.dropout(feed_forward_output) #see paper section 5.4, dropout after every sub-layer
        x = self.norm(x+feed_forward_output)
        
        return x


##################### PLACEHOLDER CLASSES TO CHAIN MULTIPLE ENCODER (DECODER) BLOCKS #####################

class Encoder(nn.Module): 

    def __init__(self, features:int, layers: nn.ModuleList): 

        """
        Placeholder class to chain EncoderBlock objects together to create the Encoder.

        Init Args:
            features    : number of features. Usually should be equal to embedding size d_model.
            layers      : ModuleList of EncoderBlock objects.
        
        """
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, features:int, layers: nn.ModuleList):

        """
        Placeholder class to chain DecoderBlock objects together to create the Decoder.

        Init Args:
            features    : number of features. Usually should be equal to embedding size d_model.
            layers      : ModuleList of DecoderBlock objects.
        
        """

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
        
        """
        Placeholder class to chain together the embedding, positional encoding, Encoder, Decoder, and projection layer.
        Contains methods to actually put the input sentence through the network.

        Init Args:
            encoder     : an instance of Encoder class
            decoder     : an instance of Decoder class
            src_embed   : an instance of InputEmbeddings class (i.e. embedding applied to input for the Encoder side)
            tgt_embed   : an instance of InputEmbeddings class (i.e. embedding applied to input for the Decoder side)
            src_pos     : an instance of PositionalEncoding class, applied to input of Encoder
            tgt_pos     : an instance of PositionalEncoding class, applied to input of Decoder
            output_layer: an instance of OutputLayer class
        """
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.output_layer = output_layer
    

    def encode(self, src, src_mask):

        """
        Puts the input sentence through the Encoder side of the network.

        Args:
            src         : input sentence.
            src_mask    : mask to be applied to encoder input.

        Returns:
            enc_output  : Encoder output.

        """

        # (batch, seq_len, d_model)
        src = self.src_embed(src) #input embedding
        src = self.src_pos(src) #positional encoding
        enc_output = self.encoder(src, src_mask) #encoder output
        return  enc_output
    

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        
        """
        Puts the input sentence (and the Encoder output) through the Decoder side of the network.

        Args:
            encoder_output  : Encoder output.
            src_mask        : mask to be applied to Encoder input sentence.
            tgt             : input sentence to the Decoder.
            tgt_mask        : mask to be applied to Decoder input sentence.

        Returns:
            dec_output  : Decoder output.

        """

        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt) #output embedding
        tgt = self.tgt_pos(tgt) #positional encoding
        dec_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask) #decoder output
        return dec_output
    

    def project(self, x):

        """
        Puts the Decoder output through the linear Output layer.
        (i.e projects Decoder input into logits).

        Args:
            x           : input (should usually be Decoder output).
        
        Returns:
            proj_output : output logits.
        """

        # (batch, seq_len, vocab_size)
        proj_output = self.output_layer(x)
        return proj_output
    



def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      n_blocks: int=6, n_heads: int=8,  
                      d_ff: int=2048, dropout_rate: float= None):
    
    """
    Assembles the previously defined modules into the Transformer network.
    See Figure 1 in the paper.

    Args:
        src_vocab_size  : vocabulary size for the Encoder input
        tgt_vocab_size  : vocabulary size for the Decoder input
        src_seq_len     : sentence length of Encoder input (i.e. define a max length, then pad sentences of less than this length) 
        tgt_seq_len     : sentence length of Decoder input (i.e. define a max length, then pad sentences of less than this length) 
        n_blocks        : number of 

        
    Returns:
        transformer     : a Transformer object.
    
    """
    
    ### Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    ### Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout_rate)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout_rate)

    ### Create the encoder blocks
    encoder_blocks = []
    for _ in range(n_blocks):
        encoder_self_attention_block = MultiHeadAttention(d_model, n_heads)
        feed_forward_block = FeedForwardBlock(d_model, d_ff)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, 
                                     feed_forward_block, dropout_rate)
        encoder_blocks.append(encoder_block)

    ### Create the decoder blocks
    decoder_blocks = []
    for _ in range(n_blocks):
        decoder_self_attention_block = MultiHeadAttention(d_model, n_heads)
        decoder_cross_attention_block = MultiHeadAttention(d_model, n_heads)
        feed_forward_block = FeedForwardBlock(d_model, d_ff)
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


