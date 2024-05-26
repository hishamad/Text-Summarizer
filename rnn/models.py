import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    Encodes a batch of source sentences. 
    """
    def __init__(
            self, 
            no_of_input_symbols, 
            embeddings=None, 
            embedding_size=16, 
            hidden_size=25,
            encoder_bidirectional=False, 
            use_gru=False, 
            tune_embeddings=False,
            device='cpu'
        ):
        
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.is_bidirectional = encoder_bidirectional
        self.embedding = nn.Embedding(no_of_input_symbols,embedding_size)
        if embeddings !=  None:
            self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float), requires_grad=tune_embeddings)
        if use_gru:
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=self.is_bidirectional)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True, bidirectional=self.is_bidirectional)
        self.device = device
        self.to(device)

    def set_embeddings(self, embeddings):
        self.embedding.weight = torch.tensor(embeddings, dtype=torch.float)

    def forward(self, x):
        """
        x is a list of lists of size (batch_size,max_seq_length)
        Each inner list contains word IDs and represents one sentence.
        The whole list-of-lists represents a batch of sentences.
       
        Returns:
        the output from the encoder RNN: a pair of two tensors, one containing all hidden states, and one 
        containing the last hidden state (see https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        """
        x_tensor = torch.tensor(x).to(self.device)
        embedded_words = self.embedding(x_tensor)
        all_hidden, last_hidden = self.rnn(embedded_words)
        return all_hidden, last_hidden


class DecoderRNN(nn.Module):
    def __init__(
            self, 
            no_of_output_symbols, 
            embedding_size=16, 
            hidden_size=25, 
            use_attention=True,
            use_gru=False,
            device='cpu'
        ):
        
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(no_of_output_symbols,embedding_size)
        self.no_of_output_symbols = no_of_output_symbols
        self.W = nn.Parameter(torch.rand(hidden_size, hidden_size)-0.5) # shouldn't W be 2*hidden_size
        self.U = nn.Parameter(torch.rand(hidden_size, hidden_size)-0.5)
        self.v = nn.Parameter(torch.rand(hidden_size, 1)-0.5)
        self.use_attention = use_attention
        if use_gru:
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, no_of_output_symbols)
        self.device = device
        self.to(device)

    def forward(self, inp, hidden, encoder_outputs):
        """
        'input' is a list of length batch_size, containing the current word
        of each sentence in the batch

        'hidden' is a tensor containing the last hidden state of the decoder, 
        for each sequence in the batch
        hidden.shape = (1, batch_size, hidden_size)

        'encoder_outputs' is a tensor containing all hidden states from the
        encoder (used in problem c)
        encoder_outputs.shape = (batch_size, max_seq_length, hidden_size)

        Note that 'max_seq_length' above refers to the max_seq_length
        of the encoded sequence (not the decoded sequence).

        Returns:
        If use_attention and display_attention are both True (task (c)), return a triple
        (logits for the predicted next word, hidden state, attention weights alpha)

        Otherwise (task (b)), return a pair
        (logits for the predicted next word, hidden state).
        """
        
        inp_tensor = torch.tensor(inp).to(self.device)
        word_embs = self.embedding(inp_tensor).unsqueeze(1)

        if not self.use_attention:
            rnn_output, hidden = self.rnn(word_embs, hidden)
            logits = self.output(rnn_output.squeeze(1))
            return logits, hidden
        
        context, alpha_ij = self.get_context(word_embs, encoder_outputs)
        rnn_output, hidden = self.rnn(word_embs, context)
        logits = self.output(rnn_output.squeeze(1))

        return logits, hidden

    def get_context(self, prev_word_embs, encoder_states):
        summed = (torch.matmul(prev_word_embs, self.U) + torch.matmul(encoder_states, self.W))
        summed = torch.tanh(summed)
        e_ij = torch.matmul(summed, self.v)
        alpha_ij = torch.softmax(e_ij, dim=1)
        context = alpha_ij * encoder_states
        context = context.sum(dim=1).unsqueeze(0)
        return context, alpha_ij


