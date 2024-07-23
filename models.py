import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, mask):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.pos_encoder, num_layers)
        self.fc = nn.Linear(d_model, 2)
    
    def forward(self, x, mask):
        embedded = self.embedding(x)
        output = self.transformer_encoder(embedded, src_key_padding_mask=~mask.bool())
        return self.fc(output.mean(dim=1))

class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 2)
    
    def forward(self, x, mask):
        outputs = self.bert(x, attention_mask=mask)
        return self.fc(outputs.pooler_output)

class GPTModel(nn.Module):
    def __init__(self):
        super(GPTModel, self).__init__()
        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(768, 2)
    
    def forward(self, x, mask):
        gpt_output = self.gpt(x, attention_mask=mask)
        pooled_output = gpt_output.last_hidden_state.mean(dim=1)
        return self.fc(pooled_output)