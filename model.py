import config
import transformers
import torch.nn as nn
import torch

# Define a class for a classification head on top of BERTBaseUncased
class BERTBaseUncasedClfHead(nn.Module):
    def __init__(self):
        super(BERTBaseUncasedClfHead, self).__init__()
        # Load the pretrained BERT model from the specified path in the config
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        
        # Add a new dense (fully connected) layer after BERT's output
        self.dense = nn.Linear(768, 512)  # 768 is BERT's hidden size, 512 is the output size

        # Layer normalization to stabilize the learning process
        self.layer_norm = nn.LayerNorm(512)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)

        # Output layer for classification (3 classes)
        self.out = nn.Linear(512, 3)

    def freeze_base_model(self):
        # Method to freeze the parameters of the BERT model to prevent them from being updated during training
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, ids, mask, token_type_ids):
        # Forward pass through BERT model
        _, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        # Pass BERT's output through the additional dense layer
        x = self.dense(pooled_output)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply dropout
        x = self.dropout(x)

        # Pass through the output layer to get final logits
        output = self.out(x)
        return output

# Define a class for a classifier that uses RoBERTa model with a GRU layer
class RobertaGRUClassifier(nn.Module):
    def __init__(self):
        super(RobertaGRUClassifier, self).__init__()
        # Load the pretrained RoBERTa model
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base')

        # GRU layer for processing sequence data
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=1, batch_first=True)

        # Batch normalization for stabilizing learning
        self.batch_norm = nn.BatchNorm1d(256)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)

        # Flatten layer to reshape data for dense layer
        self.flatten = nn.Flatten()

        # Dense layers for final classification
        self.dense1 = nn.Linear(256, 128)  # Reduce dimension from 256 to 128
        self.dense2 = nn.Linear(128, 3)    # Final classification layer, 3 classes

    def freeze_base_model(self):
        # Method to freeze the parameters of the RoBERTa model
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, last_n_layers):
        # Method to unfreeze the last `last_n_layers` layers of the RoBERTa model
        # Freeze all layers first
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers
        for layer in self.roberta.encoder.layer[-last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Forward pass through the RoBERTa model
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state

        # Pass the output through the GRU layer
        gru_output, _ = self.gru(sequence_output)
        gru_last_output = gru_output[:, -1, :]

        # Apply batch normalization if batch size is greater than 1
        if gru_last_output.size(0) > 1:
            normalized_output = self.batch_norm(gru_last_output)
        else:
            normalized_output = gru_last_output

        # Apply dropout
        dropout_output = self.dropout(normalized_output)

        # Flatten the output for dense layer input
        flattened_output = self.flatten(dropout_output)

        # Pass through dense layers
        dense_output = torch.relu(self.dense1(flattened_output))
        logits = self.dense2(dense_output)

        return logits
