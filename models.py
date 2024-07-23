import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model
import math
from sklearn.linear_model import LogisticRegression

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
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x, mask):
        # x shape: [batch_size, seq_len]
        # mask shape: [batch_size, seq_len]
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # Convert to [seq_len, batch_size, embedding_dim]
        
        # Create a boolean mask where True values are positions to be masked
        src_key_padding_mask = (mask == 0).to(x.device)
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # Convert back to [batch_size, seq_len, embedding_dim]
        
        # Use mean pooling
        output = output.mean(dim=1)
        
        return self.fc(output)

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
