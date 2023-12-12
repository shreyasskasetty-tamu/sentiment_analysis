import constants
import transformers
import torch.nn as nn
import torch

# Define a class for a classification head on top of BERTBaseUncased
class RobertaSimpleClassifier(nn.Module):
    def __init__(self):
        super(RobertaSimpleClassifier, self).__init__()
        # RoBERTa model
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base')

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(768)  # Updated to match RoBERTa output size

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Dense layer for classification
        self.classifier = nn.Linear(768, 3)  # Assuming 3 classes for classification

    def freeze_base_model(self):
        # Freeze all parameters in the RoBERTa model
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, last_n_layers):
        # Freeze all layers first
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Unfreeze the last `last_n_layers`
        for layer in self.roberta.encoder.layer[-last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # RoBERTa outputs
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # Use the <CLS> token for classification

        # Apply batch normalization only if batch size > 1
        if pooled_output.size(0) > 1:
            normalized_output = self.batch_norm(pooled_output)
        else:
            normalized_output = pooled_output

        dropout_output = self.dropout(normalized_output)

        # Classification
        logits = self.classifier(dropout_output)

        return logits


class RobertaGRUClassifier(nn.Module):
    def __init__(self):
        super(RobertaGRUClassifier, self).__init__()
        # RoBERTa model
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base')

        # GRU layer
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=1, batch_first=True)

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(256)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Dense layers
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 3)  # Assuming 3 classes for classification

    def freeze_base_model(self):
        # Freeze all parameters in the BERT model
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, last_n_layers):
        # Freeze all layers first
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Unfreeze the last `last_n_layers`
        for layer in self.roberta.encoder.layer[-last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # RoBERTa outputs
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state

        # GRU
        gru_output, _ = self.gru(sequence_output)
        gru_last_output = gru_output[:, -1, :]

        # Apply batch normalization only if batch size > 1
        if gru_last_output.size(0) > 1:
            normalized_output = self.batch_norm(gru_last_output)
        else:
            normalized_output = gru_last_output

        dropout_output = self.dropout(normalized_output)

        # Flatten the output for the dense layer
        flattened_output = self.flatten(dropout_output)

        # Dense layers
        dense_output = torch.relu(self.dense1(flattened_output))
        logits = self.dense2(dense_output)

        return logits