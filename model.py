import config
import transformers
import torch.nn as nn


class BERTBaseUncasedClfHead(nn.Module):
    def __init__(self):
        super(BERTBaseUncasedClfHead, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        # New dense layer
        self.dense = nn.Linear(768, 512)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(512)

        # Dropout for the new dense layer
        self.dropout = nn.Dropout(0.3)

        # Output layer remains the same
        self.out = nn.Linear(512, 3)
    
    def freeze_base_model(self):
        # Freeze all parameters in the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False
            

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        # Passing through the new dense layer
        x = self.dense(pooled_output)

        # Applying layer normalization
        x = self.layer_norm(x)

        # Applying dropout
        x = self.dropout(x)

        # Final output layer
        output = self.out(x)
        return output


class RoBERTaGRUModel(nn.Module):
    def __init__(self, gru_hidden_size=512, num_classes=3):
        super(RoBERTaGRUModel, self).__init__()
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base')

        # GRU Layer
        # RoBERTa outputs 768 features for each token in the sequence
        self.gru = nn.GRU(input_size=768, hidden_size=gru_hidden_size, num_layers=1, batch_first=True)

        # Batch Normalization and Dropout can still be applied here
        self.batch_norm = nn.BatchNorm1d(gru_hidden_size)
        self.dropout = nn.Dropout(0.5)

        # Adjust the input size of the first dense layer to match the GRU output
        self.dense1 = nn.Linear(gru_hidden_size, 256)
        self.dense2 = nn.Linear(256, num_classes)  # Adjust num_classes as necessary
    
    def freeze_base_model(self):
        # Freeze all parameters in the BERT model
        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Get the sequence output from RoBERTa
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state

        # GRU layer expects inputs of shape (batch, seq_len, input_size)
        gru_output, _ = self.gru(sequence_output)
        
        # Applying batch normalization and dropout to the output of the last time step
        gru_last_output = gru_output[:, -1, :]
        
        # Apply batch normalization only if batch size > 1
        if gru_last_output.size(0) > 1:
            normalized_output = self.batch_norm(gru_last_output)
        else:
            normalized_output = gru_last_output 

        dropout_output = self.dropout(normalized_output)
        
        #Dense layers for classification
        dense_output = self.dense1(dropout_output)
        logits = self.dense2(dense_output)

        return logits

# class RoBERTaGRUModel(nn.Module):
#     def __init__(self):
#         super(RoBERTaGRUModel, self).__init__()
#         self.roberta = RobertaModel.from_pretrained('roberta-base')

#         # Example of adding dropout after RoBERTa
#         self.dropout1 = nn.Dropout(0.5)

#         # GRU Layer
#         self.gru = nn.GRU(input_size=768, hidden_size=512, num_layers=1, batch_first=True)

#         # Example of adding batch normalization after GRU
#         self.batch_norm = nn.BatchNorm1d(512)

#         # Flatten layer
#         self.flatten = nn.Flatten()

#         # Dense layers with dropout before the final dense layer
#         self.dense1 = nn.Linear(512, 256)
#         self.dropout2 = nn.Dropout(0.5)
#         self.dense2 = nn.Linear(256, 3)  # Assuming 3 classes for output

#     def forward(self, input_ids, attention_mask):
#         _, pooled_output = self.roberta(input_ids, attention_mask=attention_mask, return_dict=False)
#         dropped = self.dropout1(pooled_output)
#         gru_output, _ = self.gru(dropped.unsqueeze(0))
#         normalized = self.batch_norm(gru_output.squeeze(0))
#         flattened_output = self.flatten(normalized)
#         dense_output = self.dense1(flattened_output)
#         dropped = self.dropout2(dense_output)
#         logits = self.dense2(dropped)
#         return logits
        
# class RoBERTaGRUModel(nn.Module):
#     def __init__(self):
#         super(RoBERTaGRUModel, self).__init__()
#         self.roberta = transformers.RobertaModel.from_pretrained('roberta-base')
#         self.gru = nn.GRU(input_size=768, hidden_size=512, num_layers=1, batch_first=True)
#         self.flatten = nn.Flatten()
#         self.dense1 = nn.Linear(512, 256)  # Adjust sizes as necessary
#         self.dense2 = nn.Linear(256, 3)    # Assuming 3 classes for output
#         self.softmax = nn.Softmax(dim=1)

#     def freeze_base_model(self):
#         # Freeze all parameters in the BERT model
#         for param in self.roberta.parameters():
#             param.requires_grad = False
            
#     def forward(self, input_ids, attention_mask):
#         # _, pooled_output = self.roberta(input_ids, attention_mask=attention_mask, return_dict=False)
#         # print("Shape after RoBERTa:", pooled_output.shape)  # Should be [batch_size, 768]

#         # gru_output, _ = self.gru(pooled_output.unsqueeze(0))
#         # print("Shape after GRU:", gru_output.shape)  # Check this shape

#         # flattened_output = self.flatten(gru_output)
#         # print("Shape after Flatten:", flattened_output.shape)  # Check this shape

#         # dense_output = self.dense1(flattened_output)
#         # print("Shape after Dense1:", dense_output.shape)  # Check this shape

#         # logits = self.dense2(dense_output)
#         # return self.softmax(logits)
#         sequence_output, _ = self.roberta(input_ids, attention_mask=attention_mask, return_dict=False)
    
#         # Here, you might want to use sequence_output instead of pooled_output
#         # For example:
#         gru_output, _ = self.gru(sequence_output)
#         # Use the last hidden state for classification
#         last_hidden_state = gru_output[:, -1, :]

#         dense_output = self.dense1(last_hidden_state)
#         logits = self.dense2(dense_output)
#         return self.softmax(logits)