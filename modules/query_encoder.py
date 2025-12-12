import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBertQueryEncoder(nn.Module):
    
    def __init__(self, embed_dim=128):
        super(DistilBertQueryEncoder, self).__init__()
        
        # 1. Load DistilBERT tokenizer & model
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # 2. Projection layer
        self.projection = nn.Linear(768, embed_dim)
    
    def forward(self, text_list):
        # 1. Tokenize
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"]         # shape: [batch_size, seq_len]
        attention_mask = encoded["attention_mask"]  # shape: [batch_size, seq_len]

        # 2. Pass through DistilBERT
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        
        # 3. Extract the CLS token embedding
        last_hidden_state = outputs.last_hidden_state
        cls_token_emb = last_hidden_state[:, 0, :]  # shape: [batch_size, 768]

        # 4. Project from 768 -> embed_dim 
        query_emb = self.projection(cls_token_emb)  # shape: [batch_size, embed_dim]


        return query_emb