# app.py

import streamlit as st
import torch
import torch.nn.functional as F

# Load your pre-trained PyTorch model (replace with your own model)
# Example: model = torch.load("sentiment_model.pth")
from typing import Sequence
from functools import partial
import random
import numpy as np
import os

# @title
# DO NOT CHANGE HERE
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

# Use this for getting x label
def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        yield [random.randint(0, 4) for _ in range(seq_len)]

# Use this for getting y label
def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        # note that seq is a string, not a list
        if dimer == "CG":
            cgs += 1
    return cgs

# Alphabet helpers
alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(5))}
int2dna = { i: a for a, i in zip(alphabet, range(5))}

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

LSTM_HIDDEN = 20
LSTM_LAYER = 2
batch_size = 4
learning_rate = 1e-5
epoch_num = 1000
weight_decay = 1e-3


# Model
class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self, input_size=1, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYER, output_size=1):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.classifier = torch.nn.Linear(self.hidden_size, self.output_size)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        # TODO complete forward function
        b, t = x.shape      # x -> b, t
        # x = x.unsqueeze(2)  # x -> b, t, f
        x = torch.nn.functional.one_hot(x, num_classes=self.input_size) # x -> b, t, f=5

        h0 = torch.zeros(self.num_layers, b, self.hidden_size)
        c0 = torch.zeros(self.num_layers, b, self.hidden_size)

        l1, _ = self.lstm(x.float().contiguous(), (h0,c0))
        logits = self.classifier(l1[:, -1, :])

        return logits
    
    
model = CpGPredictor(input_size=5, 
                    hidden_size=LSTM_HIDDEN, 
                    num_layers=LSTM_LAYER, 
                    output_size=1)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                            lr = learning_rate,
                            weight_decay = weight_decay)

chkpoints = ['/workspaces/streamlit-101/cpg/checkpoint.pth.tar', 
             '/workspaces/streamlit-101/cpg/best_checkpoint_relu_1hot.pth.tar',
             '/workspaces/streamlit-101/cpg/best_checkpoint_pad.pth.tar']
chkpoint_file = chkpoints[1]
if os.path.exists(chkpoint_file):
    state_dict = torch.load(chkpoint_file)
    start_epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])

############################################################

st.title("CpG Dectector App by Nayem")

# Input text
user_input = st.text_input("Enter a DNA sequence {N,A,C,G,T}:", "")

if user_input:
    # Preprocess the input (tokenization, padding, etc.)
    # Example: preprocess_input(user_input)

    # Make predictions using your model
    # Example: output = model(preprocessed_input)
    # Assume output is a probability score (e.g., [0.8, 0.2])
    model.eval()
    with torch.no_grad():
        temp = [list(dnaseq_to_intseq(seq)) for seq in user_input]
        temp = torch.tensor(temp).flatten().unsqueeze(0)
        output = model(temp).item()

    # Convert probability to sentiment label
    # sentiment_label = "Positive" if output[0] >= 0.5 else "Negative"

    st.write(f"Predicted CG count: {output}")
