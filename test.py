import config
import dataset
import warnings
import torch
import os
import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm
from engine import Engine
from model import BERTBaseUncasedClfHead, RoBERTaGRUModel
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run(model_path,model_type):

    state_dict = torch.load(model_path)
    if model_type.lower() == 'bert':
        print("Picking Bert-Uncased Model")
        model = BERTBaseUncasedClfHead()
    elif model_type.lower() == 'roberta':
        print("Picking Robert-GRU Model")
        model = RoBERTaGRUModel()
    else:
        print("Model not supported")
        sys.exit(1)
    model.load_state_dict(state_dict)

    warnings.filterwarnings('ignore')
    #Read the training dataset from the CSV file
    test_raw_dataset = pd.read_csv(config.TEST_FILE).fillna("none")

    #Preprocess Data
    processor = dataset.DatasetPreprocessor(test_raw_dataset)
    df_test = processor.preprocess_dataset()

    print(len(df_test))
    df_test = df_test.reset_index(drop=True)

    if model_type.lower() == 'bert':
        test_dataset = dataset.BERTDataset(
            review=df_test.review.values, target=df_test.sentiment.values
        )

    elif model_type.lower() == 'roberta':
        test_dataset = dataset.RoBERTaDataset(
            review=df_test.review.values, target=df_test.sentiment.values
        )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=4
    )
    
    engine = Engine(model, config.DEVICE,df_test.sentiment.values,model_type.lower())
    engine.test_eval_fn(test_data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('-m','--model_path', type=str, required=True, help='Path to the model file.')
    parser.add_argument('-t','--model_type', type=str, required=True, help='Model Name')
    args = parser.parse_args()
    run(args.model_path,args.model_type)
