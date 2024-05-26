from datetime import datetime
import argparse
import random
import pickle
import codecs
import json
import os
import nltk
import torch
import numpy as np
from pprint import pprint
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from terminaltables import AsciiTable
from tqdm import tqdm
import json
import pandas as pd
from torchmetrics.text.rouge import ROUGEScore
from functools import partial
import logging

from config import Config

from models import *

logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[
                        logging.FileHandler("log.txt"),
                        logging.StreamHandler()
                    ])

class AmazonDataset(Dataset):
    """
    A dataset with source sentences and their respective translations
    into the target language.

    Each sentence is represented as a list of word IDs. 
    """
    def __init__(self, data, record_symbols=True):
        try:
            nltk.word_tokenize("hi there.")
        except LookupError:
            nltk.download('punkt')
        self.source_list = []
        self.target_list = []
        # Read the datafile
        
        for i in tqdm(range(len(data))):
            s = data.Text[i]
            t = data.Summary[i]
            source_sentence = []
            for w in nltk.word_tokenize(s):
                w = w.lower()
                if w not in source_i2w and record_symbols:
                    source_w2i[w] = len(source_i2w)
                    source_i2w.append(w)
                source_sentence.append(source_w2i.get(w, source_w2i[UNK_SYMBOL]))
            source_sentence.append(source_w2i[END_SYMBOL])
            self.source_list.append(source_sentence)
            target_sentence = []
            for w in nltk.word_tokenize(t):
                w = w.lower()
                if w not in target_i2w and record_symbols:
                    target_w2i[w] = len(target_i2w)
                    target_i2w.append(w)
                target_sentence.append(target_w2i.get(w, target_w2i[UNK_SYMBOL]))
            target_sentence.append(target_w2i[END_SYMBOL])
            self.target_list.append(target_sentence)

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, idx):
        return self.source_list[idx], self.target_list[idx]


def load_data():
    source_w2i = {}
    source_i2w = []
    target_w2i = {}
    target_i2w = []
    
    with open('data/source_w2i.json', 'r') as f:
        source_w2i = json.load(f)
    
    with open('data/source_i2w.json', 'r') as f:
        source_i2w = json.load(f)
    
    with open('data/target_w2i.json', 'r') as f:
        target_w2i = json.load(f)
    
    with open('data/target_i2w.json', 'r') as f:
        target_i2w = json.load(f)

    return source_w2i, source_i2w, target_w2i, target_i2w
    

def pad_sequence(batch, pad_source, pad_target):
    source, target = zip(*batch)
    max_source_len = max(map(len, source))
    max_target_len = max(map(len, target))
    padded_source = [[b[i] if i < len(b) else pad_source for i in range(max_source_len)] for b in source]
    padded_target = [[l[i] if i < len(l) else pad_target for i in range(max_target_len)] for l in target]
    return padded_source, padded_target

def load_glove_embeddings(embedding_file, source_w2i, source_i2w):
    """
    Reads pre-made embeddings from a file
    """
    N = len(source_w2i)
    embeddings = [0]*N
    with codecs.open(embedding_file, 'r', 'utf-8') as f:
        for line in f:
            data = line.split()
            word = data[0].lower()
            if word not in source_w2i:
                source_w2i[word] = N
                source_i2w.append(word)
                N += 1
                embeddings.append(0)
            vec = [float(x) for x in data[1:]]
            D = len(vec)
            embeddings[source_w2i[word]] = vec
    # Add a '0' embedding for the padding symbol
    embeddings[0] = [0]*D
    # Check if there are words that did not have a ready-made Glove embedding
    # For these words, add a random vector
    for word in source_w2i:
        index = source_w2i[word]
        if embeddings[index] == 0:
            embeddings[index] = (np.random.random(D)-0.5).tolist()
    return D, embeddings, source_w2i, source_i2w

def test(config, encoder, decoder, test_dataset, target_w2i, target_i2w):
    encoder.eval()
    decoder.eval()

    num_correct_words = 0
    num_correct_sentences = 0
    
    tot_words = 0
    tot_sentances = 0

    predicted_sentences = []
    correct_sentences = []
    
    for x, y in test_dataset:
        predicted_sentence = []
        outputs, hidden = encoder([x])
        if encoder.is_bidirectional:
            hidden = hidden.permute((1,0,2)).reshape(1,-1).unsqueeze(0)
        
        predicted_symbol = target_w2i[config.START_SYMBOL]
        predicted_sentence = []
        num_attempts = 0
        while num_attempts < config.MAX_PREDICTIONS:
            predictions, hidden = decoder([predicted_symbol], hidden, outputs)    
            _, predicted_tensor = predictions.topk(1)
            predicted_symbol = predicted_tensor.detach().item()
    
            num_attempts += 1
    
            if predicted_symbol == target_w2i[config.END_SYMBOL]:
                break
                
            predicted_sentence.append(predicted_symbol)

        # [:-1] such we dont consider the end symbol
        y = y[:-1]
        
        if predicted_sentence == y:
            num_correct_sentences += 1

        for w_p, w_y in zip(predicted_sentence, y):
            if w_p == w_y:
                num_correct_words += 1

        tot_words += len(y)
        tot_sentances += 1

        predicted_sentence_str = " ".join([target_i2w[i] for i in predicted_sentence])
        correct_sentence_str = " ".join([target_i2w[i] for i in y])

        predicted_sentences.append(predicted_sentence_str)
        correct_sentences.append(correct_sentence_str)

    rouge = ROUGEScore()
    r = rouge(predicted_sentences, correct_sentences)
    for key, item in r.items():
        logging.info(f"{key}: {item}")

    logging.info('')

    word_acc = num_correct_words / tot_words
    sent_acc = num_correct_sentences / tot_sentances

    logging.info(f"Word acc: {word_acc*100:.2f}%")
    logging.info(f"Sent acc: {sent_acc*100:.2f}%")


def train(
        config, 
        train_dataset, 
        test_dataset,
        embeddings,
        embedding_size,
        source_i2w,
        source_w2i,
        target_i2w,
        target_w2i
    ):
    
    
    encoder = EncoderRNN(
        len(source_i2w),
        embeddings=embeddings,
        embedding_size=embedding_size,
        hidden_size=config.hidden_size,
        encoder_bidirectional=config.bidirectional,
        tune_embeddings=config.tune_embeddings,
        use_gru=config.use_gru,
        device=config.device
    )
    decoder = DecoderRNN(
        len(target_i2w),
        embedding_size=embedding_size,
        hidden_size=config.hidden_size*(config.bidirectional+1),
        use_attention=config.use_attention,
        use_gru=config.use_gru,
        device=config.device
    )

    
    pad_sequence_partial = partial(
        pad_sequence, 
        pad_source=source_w2i[config.PADDING_SYMBOL], 
        pad_target=target_w2i[config.PADDING_SYMBOL]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_sequence_partial)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_sequence_partial)

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(
        encoder_optimizer, 
        step_size=1, 
        gamma=0.85
    )
    
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(
        decoder_optimizer, 
        step_size=1, 
        gamma=0.85
    )

    for epoch in range(config.epochs):
        encoder.train()
        decoder.train()
        for j, (source, target) in enumerate(train_loader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0
            
            outputs, hidden = encoder(source)
            if config.bidirectional:
                hidden = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=1).unsqueeze(0)
                        
            teacher_forcing_ratio = 1 # - epoch/config.epochs
            
            idx = [target_w2i[config.START_SYMBOL] for sublist in target]
            predicted_symbol = [target_w2i[config.START_SYMBOL] for sublist in target]
    
            target_length = len(target[0])
            for i in range(target_length):
                use_teacher_forcing = (random.random() < teacher_forcing_ratio)
                if use_teacher_forcing:
                    predictions, hidden = decoder(idx, hidden, outputs)
                else:
                    predictions, hidden = decoder(predicted_symbol, hidden, outputs)
                    
                _, predicted_tensor = predictions.topk(1)
                predicted_symbol = predicted_tensor.squeeze().tolist()
    
                idx = [sublist[i] for sublist in target]
                loss += criterion(predictions.squeeze(), torch.tensor(idx).to(config.device))
                
            loss /= (target_length * config.batch_size)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            print(f"\rEpoch {epoch}, loss:, {loss.item():.4f}, {j/len(train_loader)*100:.2f}%", end="")
            
        encoder_scheduler.step()
        decoder_scheduler.step()
        print()

    print()

    model_name = ""
    if config.use_gru:
        model_name += "gru"
    else:
        model_name += "rnn"

    if config.use_attention:
        model_name += "_attn"
    else:
        model_name += "_no_attn"
        
    os.makedirs("saved_models", exist_ok=True)
    torch.save(encoder.state_dict(), f"saved_models/encoder_{model_name}.pt")
    torch.save(decoder.state_dict(), f"saved_models/decoder_{model_name}.pt")

    test(
        config,
        encoder, 
        decoder,
        test_dataset, 
        target_w2i,
        target_i2w
    )


def run_experiments():
    train_dataset = torch.load('data/amazon_train_dataset_py')
    test_dataset = torch.load('data/amazon_test_dataset_py')

    source_w2i, source_i2w, target_w2i, target_i2w = load_data()

    embedding_size, embeddings, source_w2i, source_i2w = load_glove_embeddings(
        '/datasets/dd2417/glove.6B.50d.txt',
        source_w2i=source_w2i,
        source_i2w=source_i2w
    )

    pure_rnn_config = Config()
    pure_rnn_config.use_gru = False
    pure_rnn_config.use_attention = False

    rnn_attn_config = Config()
    rnn_attn_config.use_gru = False
    rnn_attn_config.use_attention = True

    gru_config = Config()
    gru_config.use_gru = True
    gru_config.use_attention = False

    gru_attn_config = Config()
    gru_attn_config.use_gru = True
    gru_attn_config.use_attention = True
    
    configs = [
        pure_rnn_config,
        rnn_attn_config, 
        gru_config, 
        gru_attn_config
    ]

    for cfg in configs:
        logging.info("-"*100)
        logging.info(f"GRU: {cfg.use_gru}, Attn: {cfg.use_attention}")
        train(
            config=cfg,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            embeddings=embeddings,
            embedding_size=embedding_size,
            source_w2i=source_w2i,
            source_i2w=source_i2w,
            target_w2i=target_w2i,
            target_i2w=target_i2w
        )


if __name__ == '__main__':
    run_experiments()
        

    
    












    


