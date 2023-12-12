import constants
import dataset
import argparse
import warnings
import torch
import os
import pandas as pd
import torch.nn as nn
import numpy as np
import sys
import logging
import datetime

from config import Config
from tqdm import tqdm
from engine import Engine
from model import RobertaSimpleClassifier, RobertaGRUClassifier
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

class Trainer:
    def __init__(self, config, base_path, dataset, log_dir="./",device = 'cuda'):
        self.dataset = dataset
        self.config = config
        self.training_accuracies = []
        self.validation_accuracies = []
        self.training_losses = []
        self.base_path = base_path
        self.device = torch.device(device)
        self.log_dir = log_dir
        self.logger = self.init_logger()
        
    def save_preprocessed_data(self, data, filepath):
        data.to_csv(filepath, index=False)

    def load_preprocessed_data(self, filepath):
        return pd.read_csv(filepath)

    def init_logger(self):
        # Create a logger
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        logger = logging.getLogger(f"trainer_{self.config.model_type}_{timestamp}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            # Create handlers (console and file)
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler(f'{self.log_dir}/training_{self.config.model_type}.log')
            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.INFO)

            # Create formatters and add them to handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(formatter)
            f_handler.setFormatter(formatter)

            # Add handlers to the logger
            logger.addHandler(c_handler)
            logger.addHandler(f_handler)

        return logger

    def generate_model_path(self):
        # Construct a unique file name based on configuration parameters
        file_name = f"model_{self.config.model_type}_lr{self.config.learning_rate}_bs{self.config.train_batch_size}_ep{self.config.epochs}"
        if self.config.unfreeze_layers is not None:
            file_name += f"_unfreeze{self.config.unfreeze_layers}"

        file_name += ".pt"  # Add file extension for PyTorch model

        # Combine with the base directory
        model_path = self.base_path+file_name
        return model_path

    def train(self, epoch, train_data_loader, engine, optimizer, scheduler):
        # Training loop
        final_loss = 0
        final_accuracy = 0
        with tqdm(enumerate(train_data_loader), total=len(train_data_loader), unit="Batch") as data_loader_tqdm:
            data_loader_tqdm.set_description(f"Epoch {epoch}")
            train_loss, train_accuracy = engine.train_fn(data_loader_tqdm, optimizer, scheduler)
            self.training_accuracies.append(train_accuracy)
            self.training_losses.append(train_loss)
        self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    def evaluate(self, epoch, valid_data_loader, engine, best_accuracy, model):

        with tqdm(enumerate(valid_data_loader), total=len(valid_data_loader)) as data_loader_tqdm:
            outputs, targets = engine.eval_fn(data_loader_tqdm)

        # Convert outputs to numpy arrays and then to class indices
        predicted_labels = np.argmax(outputs, axis=1)
        # Convert targets to numpy arrays
        targets = np.array(targets)

        accuracy = metrics.accuracy_score(targets, predicted_labels)
        self.validation_accuracies.append(accuracy)
        self.logger.info(f"Validation - Epoch: {epoch} Accuracy: {100. * accuracy}%")
        if accuracy > best_accuracy:
            model_path = self.generate_model_path()
            print(model_path)
            model_dir = os.path.dirname(model_path)

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_path)
            best_accuracy = accuracy

    def run(self):
        warnings.filterwarnings('ignore')

        df_train, df_valid = model_selection.train_test_split(
            self.dataset, test_size=0.1, random_state=42, stratify=self.dataset.sentiment.values
        )

        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        train_dataset = dataset.RoBERTaDataset(
            review=df_train.review.values, target=df_train.sentiment.values
        )
        valid_dataset = dataset.RoBERTaDataset(
            review=df_valid.review.values, target=df_valid.sentiment.values
        )


        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.train_batch_size, num_workers=4
        )

        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.config.valid_batch_size, num_workers=1
        )

        if self.config.model_type.lower() == 'roberta-simple':
            print("Picking Roberta Simple Classifier Model")
            model = RobertaSimpleClassifier()
        elif self.config.model_type.lower() == 'roberta-gru':
            print("Picking Robert-GRU Model")
            model = RobertaGRUClassifier()
        else:
            print("Model not supported")
            sys.exit(1)

        model.to(self.device)
        model.freeze_base_model()
        param_optimizer = list(model.named_parameters())

        num_train_steps = int(len(df_train) / self.config.train_batch_size * self.config.epochs)
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        engine = Engine(model,self.device,df_train.sentiment.values,self.config.model_type.lower())
        best_accuracy = 0

        for epoch in range(self.config.epochs):
            self.train(epoch, train_data_loader, engine, optimizer, scheduler)
            self.evaluate(epoch, valid_data_loader, engine, best_accuracy, model)

        if self.config.unfreeze_layers:
            model.unfreeze_layers(self.config.unfreeze_layers)
            print(f"Unfreezing last {self.config.unfreeze_layers} layers of RoBERTa.")
            optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
            )
            for epoch in range(8):
                self.train(epoch, train_data_loader, engine, optimizer, scheduler)
                self.evaluate(epoch, valid_data_loader, engine, best_accuracy,model)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter a model\n1. roberta-simple\n2. Bert')
    parser.add_argument('-m','--model', type=str, required=True, help='Model Name')
    args = parser.parse_args()
    if args.model is not None:
        balanced_preprocessed_dataset = pd.read_csv(constants.BALANCED_DATASET_PATH)
        config = Config(3e-5, 32,64, 4, 'roberta-gru')
        config.pretty_print()
        trainer = Trainer(config,constants.BASE_MODEL_PATH,balanced_preprocessed_dataset)
        trainer.run()
