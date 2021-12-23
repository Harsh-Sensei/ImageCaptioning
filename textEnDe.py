import random

import torch
import torch.nn as nn  # Neural networks like fully connected layers, CNNs, RNNs, LSTMs, GRUs
import torch.optim as optim  # Optimiations like SGD, Adam
import torch.nn.functional as F  # For activation functions like RELU, Tanh, sigmoid
from torch.utils.data import DataLoader  # For dataset management

import torchvision.datasets as datasets  # Standard datasets, has COCO
import torchvision.transforms as transforms  # transformations on dataset
import torchvision.models as models
#from CustomDatasets import Sydney_Captions, UCM_Captions  # Datasets involving captions
import numpy as np
import spacy
#from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence,bleu,save_checkpoint,load_checkpoint

#field
from torchtext.legacy.data import Field, BucketIterator

#dataset
from torchtext.datasets import Multi30k



# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#
# # loading the UCM_Captions and Sydney_Captions datasets
# UCM_dataset = UCM_Captions(ret_type="caption-caption")
# Sydney_dataset = Sydney_Captions(ret_type="caption-caption")
#
# UCM_len = UCM_dataset.__len__()
# Sydney_len = Sydney_dataset.__len__()
#
# # Setting up training and testing data
# UCM_train_set, UCM_test_set = torch.utils.data.random_split(UCM_dataset,
#                                                             [int(UCM_len * 0.8), UCM_len - int(UCM_len * 0.8)])
# Sydney_train_set, Sydney_test_set = torch.utils.data.random_split(UCM_dataset, [int(Sydney_len * 0.8),
#                                                                                 Sydney_len - int(Sydney_len * 0.8)])
#
# # Initializing dataloader
# UCM_train_loader = DataLoader(dataset=UCM_train_set, batch_size=batch_size, shuffle=True)
# UCM_test_loader = DataLoader(dataset=UCM_test_set, batch_size=batch_size, shuffle=True)
# Sydney_train_loader = DataLoader(dataset=Sydney_train_set, batch_size=batch_size, shuffle=True)
# Sydney_test_loader = DataLoader(dataset=Sydney_test_set, batch_size=batch_size, shuffle=True)



spacy_eng=spacy.load("en_core_web_sm")

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

english=Field(tokenize=tokenizer_eng,lower=True,init_token='<sos>',eos_token='<eos>')

train_data,validation_data,test_data=Multi30k()

english.build_vocab(train_data,max_size=10000,min_freq=2)

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,p):
        super(Encoder,self).__init__()

        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.dropout=nn.Dropout(p)  ## See
        self.embedding=nn.Embedding(input_size, embedding_size)
        self.rnn=nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)

    def forward(self,x):
        #x is the vector of indices
        #shape of x is (seq_length,N)
        embedding=self.dropout(self.embedding(x))

        outputs,(hidden,cell)=self.rnn(embedding)

        return hidden,cell


class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,p):
        super(Decoder,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.dropout=nn.Dropout(p)
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.rnn=nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p,bidirectional=True)
        self.fc=nn.Linear(hidden_size,output_size)

        def forward(self,x,hidden,cell):
            x=x.unsqueeze(0)
            embedding=self.dropout(self.embedding(x))
            outputs,(hidden,cell)=self.rnn(embedding,(hidden,cell))
            predictions=self.fc(outputs)
            predictions=predictions.squeeze(0)
            return predictions,hidden,cell


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,source,target,teacher_force_ratio=0.5):

        batch_size=source.shape[1]
        target_len=target.shape[0]
        target_vocab_size=len(english.vocab)

        outputs=torch.zeros(target_len,batch_size,target_vocab_size).to(device)

        hidden,cell=self.encoder(source)

        x=target[0]

        for t in range(1,target_len):
            output,hidden,cell=self.decoder(x,hidden,cell)
            outputs[t]=output
            best_guess=output.argmax(1)

            x=target[t] if random.random()<teacher_force_ratio else best_guess

        return outputs




# Hyper parameters
learning_rate=0.001
num_epochs=10
batch_size=32

load_model=False
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder=len(english.vocab)
input_size_decoder=len(english.vocab)
output_size=len(english.vocab)
encoder_embedding_size=300
decoder_embedding_size=300
hidden_size=1024
num_layers=2
enc_dropout=0.5
dec_dropout=0.5


#Tensorboard

#writer=SummaryWriter(f'runs/loss_plot')
step=0

train_iterator,valid_iterator,test_iterator=BucketIterator.splits((train_data,validation_data,test_data),
                                                                  batch_size=batch_size,
                                                                  sort_within_batch=True,sort_key=lambda x:len(x.src),device=device)

encoder_net=Encoder(input_size_encoder,encoder_embedding_size,hidden_size,num_layers,enc_dropout).to(device)
decoder_net=Decoder(input_size_decoder,decoder_embedding_size,hidden_size,output_size,num_layers,dec_dropout).to(device)

model=Seq2Seq(encoder_net,decoder_net).to(device)
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

criterion=nn.MSELoss()

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'),model,optimizer)

for epoch in range(num_epochs):
    print(f'Epoch [{epoch}/{num_epochs}]')

    checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for batch_idx,batch in enumerate(train_iterator):
        inp_data=batch.src.to(device)
        target=batch.trg.to(device)

        output=model(inp_data,target)

        output=output.reshape(-1,output.shape[2])
        target=target[1:].reshape(-1)

        optimizer.zero_grad()
        loss=criterion(output,target)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1)
        optimizer.step()
        #writer.add_scalar('Training Loss',loss,global_step=step)
        step+=1
