import torch
from torch.utils.data import random_split, DataLoader
from pathlib import Path

#Huggingface libraries
import datasets 
import tokenizers as tkz

#Home-made libraries
import tfr_dataset



### tokenizers guide ###
# https://huggingface.co/docs/transformers/tokenizer_summary
# https://huggingface.co/docs/tokenizers/quicktour
# https://huggingface.co/learn/nlp-course/chapter6/2?fw=pt


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #if tokenizer path exists
    if not Path.exists(tokenizer_path): #otherwise build
        tokenizer = tkz.Tokenizer(tkz.models.WordLevel(unk_token="[UNK]"))
        # instantiate Tokenizer class with desired model
        tokenizer.pre_tokenizer = tkz.pre_tokenizers.Whitespace() 
        # split input into words according to whitespace
        # so that a token do not correspond to more than one word
        trainer = tkz.trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # load the appropriate trainer corresponding to the selected tokenizer model
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = tkz.Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# note: "training" a tokenizer is a deterministic process, not stochastic like training NN.


def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = datasets.load_dataset('opus_books',f"{config['datasource']}", 
                                   f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) #build tokenizer for source language dataset
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt']) #build tokenizer for target language dataset

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) #torch.utils.data.random_split

    train_ds = tfr_dataset.BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = tfr_dataset.BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


"""

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.model import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace




def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset('opus_books',f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

"""