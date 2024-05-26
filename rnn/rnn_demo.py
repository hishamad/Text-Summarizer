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





with open('data/source_w2i.json', 'r') as f:
    source_w2i = json.load(f)

with open('data/source_i2w.json', 'r') as f:
    source_i2w = json.load(f)

with open('data/target_w2i.json', 'r') as f:
    target_w2i = json.load(f)

with open('data/target_i2w.json', 'r') as f:
    target_i2w = json.load(f)


PADDING_SYMBOL = ' '
START_SYMBOL = '<START>'
END_SYMBOL = '<END>'
UNK_SYMBOL = '<UNK>'
MAX_PREDICTIONS = 20


def load_glove_embeddings(embedding_file):
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
    return D, embeddings


def summarize(review_text, encoder, decoder):
    input_review = []
    for w in nltk.word_tokenize(review_text):
        w = w.lower()
        input_review.append(target_w2i.get(w, target_w2i[UNK_SYMBOL]))
    input_review.append(target_w2i[END_SYMBOL])


    
    predicted_summary = []
    outputs, hidden = encoder([input_review])
    hidden = hidden.permute((1,0,2)).reshape(1,-1).unsqueeze(0)
    
    predicted_symbol = target_w2i[START_SYMBOL]
    num_attempts = 0
    while num_attempts < MAX_PREDICTIONS:
        predictions, hidden = decoder([predicted_symbol], hidden, outputs)    
        _, predicted_tensor = predictions.topk(1)
        predicted_symbol = predicted_tensor.detach().item()

        num_attempts += 1

        if predicted_symbol == target_w2i[END_SYMBOL]:
            break
            
        predicted_summary.append(predicted_symbol)

    return " ".join([target_i2w[i] for i in predicted_summary])



def main():
    config = Config()
    config.use_gru = True
    config.use_attention = True

    embedding_size, embeddings = load_glove_embeddings('/datasets/dd2417/glove.6B.50d.txt')
    
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


    model_name = ""
    if config.use_gru:
        model_name += "gru"
    else:
        model_name += "rnn"

    if config.use_attention:
        model_name += "_attn"
    else:
        model_name += "_no_attn"
        
    encoder.load_state_dict(torch.load(f"saved_models/encoder_{model_name}.pt"))
    decoder.load_state_dict(torch.load(f"saved_models/decoder_{model_name}.pt"))

    encoder.eval()
    decoder.eval()

    while True:
        print("-"*40)
        print()
        inp = input("Input review: ")
        out = summarize(inp, encoder, decoder)
        print()
        print(f"Summary: {out}")


main()

    

    
    