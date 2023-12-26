import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        #turn the SOS, EOS, PAD tokens from text format into numerical index, and store as tensor
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) # Start of Sentence 
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64) # End of Sentence
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64) # Padding

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]

        #extract the source text and target text
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens (ie turn input sentence into sequence of numerical token indices)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add padding so that all input sentence have the same length (ie seq_len)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  
        # -2 because we add SOS & EOS tokens to the input sentence, so we need 2 less padding
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 
        # just -1 because we only add SOS token to decoder input
        # and only add EOS token to decoder output

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")


        # Add SOS token, EOS token, and pad to the encoder input sentence
        encoder_input = torch.cat(
                                [self.sos_token, 
                                torch.tensor(enc_input_tokens, dtype=torch.int64),
                                self.eos_token,
                                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)],
                                dim=0)
 

        # Add only SOS token and pad to the decoder input sentence
        decoder_input = torch.cat(
                                [self.sos_token,
                                torch.tensor(dec_input_tokens, dtype=torch.int64),
                                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],
                                dim=0)

        # Add only EOS token and pad to the label sentence
        label = torch.cat(
                        [torch.tensor(dec_input_tokens, dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],
                        dim=0)

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        #return a dictionary
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)

            # add encoder mask so that the attention  do not consider the padding tokens
            # also unsqeeze to add sequence dimension & batch dimension
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)

            # add decoder mask so that the decoder attention do not look at words after, so that the network doesnt "cheat"
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len) --> broadcasting
            # note: & is python bitwise AND operator

            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
    # this returns true for matrix elements below the diagonal and false for elements above the diagonal



"""
class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) #turn the token from text into numerical index
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens (ie turn input sentence into sequence of numerical token indices)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 because we add SOS & EOS tokens to the input sentence
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 
        # just -1 because we only add SOS token to decoder input
        # and only add EOS token to decoder output

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add SOS and EOS token and pad
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only SOS token and pad
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only EOS token and pad
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # add encoder mask so that the attention  do not consider the padding tokens
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # add decoder mask so that the decoder attention do not look at words after
            # so that the network doesnt cheat
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
    # this returns true for matrix elements below the diagonal and false for elements above the diagonal

"""