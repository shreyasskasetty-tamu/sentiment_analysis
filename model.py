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