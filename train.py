import config
import dataset
import argparse
import warnings
import torch
import os
import pandas as pd
import torch.nn as nn
import numpy as np
import sys

from tqdm import tqdm
from engine import Engine
from model import BERTBaseUncasedClfHead, RoBERTaGRUModel
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def save_preprocessed_data(data, filepath):
    data.to_csv(filepath, index=False)

def load_preprocessed_data(filepath):
    return pd.read_csv(filepath)

def run(model_type):
    warnings.filterwarnings('ignore')

    # Check if preprocessed data exists
    if os.path.exists(config.BALANCED_DATASET_PATH):
        print("Loading preprocessed data...")
        dfx = load_preprocessed_data(config.BALANCED_DATASET_PATH)
    else:
        print("Preprocessing data...")
        raw_data_df = pd.read_csv(config.TRAINING_FILE)
        proccessor = dataset.DatasetPreprocessor(raw_data_df)
        dfx = proccessor.preprocess_dataset()
        save_preprocessed_data(dfx, config.PREPROCESSED_DATA_PATH)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values
    )
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    if model_type.lower() == 'bert':
        train_dataset = dataset.BERTDataset(
            review=df_train.review.values, target=df_train.sentiment.values
        )
        valid_dataset = dataset.BERTDataset(
            review=df_valid.review.values, target=df_valid.sentiment.values
        )

    elif model_type.lower() == 'roberta':
        train_dataset = dataset.RoBERTaDataset(
            review=df_train.review.values, target=df_train.sentiment.values
        )
        valid_dataset = dataset.RoBERTaDataset(
            review=df_valid.review.values, target=df_valid.sentiment.values
        )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    if model_type.lower() == 'bert':
        print("Picking Bert-Uncased Model")
        model = BERTBaseUncasedClfHead()
    elif model_type.lower() == 'roberta':
        print("Picking Robert-GRU Model")
        model = RoBERTaGRUModel()
    else:
        print("Model not supported")
        sys.exit(1)

    model.to(device)
    model.freeze_base_model()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    engine = Engine(model,device,df_train.sentiment.values,model_type.lower())
    best_accuracy = 0
    
    for epoch in range(config.EPOCHS):
        # Training loop
        final_loss = 0
        final_accuracy = 0
        with tqdm(enumerate(train_data_loader), total=len(train_data_loader), unit="Batch") as data_loader_tqdm:
            data_loader_tqdm.set_description(f"Epoch {epoch}")
            train_loss, train_accuracy = engine.train_fn(data_loader_tqdm, optimizer, scheduler)

        print(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        with tqdm(enumerate(valid_data_loader), total=len(valid_data_loader)) as data_loader_tqdm:
            outputs, targets = engine.eval_fn(data_loader_tqdm)
        
        # Convert outputs to numpy arrays and then to class indices
        predicted_labels = np.argmax(outputs, axis=1)

        # Convert targets to numpy arrays
        targets = np.array(targets)

        accuracy = metrics.accuracy_score(targets, predicted_labels)
        print(f"Validation - Epoch: {epoch} Accuracy: {100. * accuracy}%")
        if accuracy > best_accuracy:
            model_path = config.MODEL_PATH
            model_dir = os.path.dirname(model_path)

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter a model\n1. Roberta\n2. Bert')
    parser.add_argument('-m','--model', type=str, required=True, help='Model Name')
    args = parser.parse_args()
    if args.model is not None:
        run(args.model)
