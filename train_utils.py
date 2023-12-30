"""
stty is not recognized error?

"""

import os
from pathlib import Path
from tqdm import tqdm #for progress bar

# torch related
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter #for visualization
import torchmetrics


#Huggingface libraries
import datasets 
import tokenizers as tkz

#Home-made libraries
import tfr_dataset
import tfr_model
import config_utils

CUDA_LAUNCH_BLOCKING = 1 #what is this for

### tokenizers guide ###
# https://huggingface.co/docs/transformers/tokenizer_summary
# https://huggingface.co/docs/tokenizers/quicktour
# https://huggingface.co/learn/nlp-course/chapter6/2?fw=pt


##### BUILDING TOKENIZER #####

# note: "training" a tokenizer is a deterministic process, not stochastic like training NN.

# def get_all_sentences_opus_books(ds,lang):
#     for item in ds:
#         yield item['translation'][lang] 
#         #returns a generator

# def get_all_sentences_indonlp(ds, lang): #indonlp/NusaX-MT
#     lang_idx = None
#     if lang == 'eng':
#         lang_idx = 1
#     elif lang == 'ind':
#         lang_idx = 2

#     for item in ds:
#         yield item[f'text_{lang_idx}'] #1 for english, 2 for indonesian

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang] 
        #returns a generator


def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #path to the tokenizer
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



##### LOAD AND TOKENIZE THE DATASET #####

def get_ds(config):
    # It only has the train split, so we divide it overselves
    #ds_raw = datasets.load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train+validation+test')
    #train_ds_raw = datasets.load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    #val_ds_raw = datasets.load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='validation')
    ds_raw = datasets.load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) #build tokenizer for source language dataset
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt']) #build tokenizer for target language dataset

    # in case the dataset has not been pre-split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) #torch.utils.data.random_split


    ### convert the raw dataset into torch Dataset object ###
    train_ds = tfr_dataset.BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = tfr_dataset.BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids #for opus_books
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        #src_ids = tokenizer_src.encode(item['text_1']).ids #using indonlp dataset
        #tgt_ids = tokenizer_src.encode(item['text_2']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



##### BUILD THE TRANSFORMER #####

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = tfr_model.build_transformer(
                                vocab_src_len, vocab_tgt_len, 
                                config['seq_len'], config['seq_len'], config['d_model'],
                                config['n_blocks'], config['n_heads'],  
                                config['d_ff'], config['dropout_rate']
                                ) #n_blocks, n_heads, d_ff, and dropout_rate should be part of config
    return model


##### TRAIN FUNCTIONS #####

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training device is {device}')

    #ensure we have the folder to save the weights
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # DataLoader and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device) #put to device

    #run tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), lr= config['lr'], eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) 
    #use label smoothing to reduce overfitting

    # preloading settings
    initial_epoch = 0 #0 if we start from beginning
    global_step = 0

    if config['preload']: #so we can resume training - need model state and optimizer state
        model_filename = config_utils.get_weights_file_path(config, epoch= config['preload'])
        print(f'preloading model {model_filename}')
        model_state = torch.load(model_filename)
        initial_epoch = model_state['epoch'] + 1
        optimizer.load_state_dict(model_state['optimizer_state_dict'])
        global_step = model_state['global_step']


    ### TRAINING LOOP ###
        
    for epoch in range(initial_epoch, config['n_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f'processing epoch {epoch:02d}') #iterator but also prints progress bar

        for batch in batch_iterator:
            model.train() #put model in training mode

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            #run the tensors through the transformer
            #reminder: encode, decode, and project are methods of the Transformer class in tfr_model library
            encoder_output = model.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(B, seq_len, d_model)
            projection_output = model.project(decoder_output) #(B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) #(B, seq_len)

            # (B, seq_len, tgt_vocab_size) --> (B*seq_len, tgt_vocab_size)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) 
            #why do we need the view method?
            #check CrossEntropyLoss
            batch_iterator.set_postfix({f'loss':f'{loss.item():6.3f}'}) #display loss after progress bar

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1


        ### validation after every epoch
        # run_validation function is provided below
        run_validation(model, val_dataloader, tokenizer_tgt, 
                       config['seq_len'], device, lambda msg: batch_iterator.write(msg), 
                       global_step, writer)


        ### Save the model at the end of every epoch
        # model_filename = config_utils.get_weights_file_path(config, f"{epoch:02d}")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'global_step': global_step}, 
        #     model_filename)
    
    #save model at the end of training   
    model_filename = config_utils.get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step}, 
            model_filename)

##### VALIDATION FUNCTIONS #####

            
def greedy_decode(model, source, source_mask,tokenizer_tgt, max_len, device): #tokenizer_src is not used?
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    # remember that in the transformer decoder, the 2nd multihead attention block takes the encoder output as K & V
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) #why not just torch.Tensor(sos_idx) ?

    # keep predicting next word until we reach max sequence length or end of sentence
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = tfr_dataset.causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate decoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token and add it to the decoder input (ie for the next run the decoder input has an additional word)
        prob = model.project(out[:, -1]) #why the index?
        _, next_word = torch.max(prob, dim=1) #take the maximum probability, ie "greedy"

        decoder_input = torch.cat(
                                [decoder_input, 
                                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
                                dim=1)

        if next_word == eos_idx:
            break #end the loop if we reach end of sentence

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    print(f'validation device is {device}')
    model.eval() #set model to eval mode
    count = 0

    source_texts = []
    expected = []
    predicted = []

    #for display purposes on the console
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad(): # do not calculate gradient for validation
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len) 
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output 
            # (print_msg is used instead of print because we use tqdm to display progress bar)
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()





##### actually running the training #####


if __name__ == '__main__':
    # config = config_utils.get_config()
    # ds_raw = datasets.load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    # get_all_sentences(ds_raw,1)
    config = config_utils.get_config()
    #print(config)
    print('starting training script')
    train_model(config)