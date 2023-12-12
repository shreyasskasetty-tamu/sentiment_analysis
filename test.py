import constants
import dataset
import warnings
import torch
import pandas as pd
import numpy as np
import argparse
import sys

from config import Config
from engine import Engine


class Tester:
  
  def __init__(self, dataset, config, device = 'cuda',base_model_path='/content/drive/MyDrive/Models/'):
      self.dataset = dataset
      self.config = config
      self.device = torch.device(device)
      self.model_path = self.generate_model_path(base_model_path)
  
  def generate_model_path(self,base_model_path):
        # Construct a unique file name based on configuration parameters
        file_name = f"model_{self.config.model_type}_lr{self.config.learning_rate}_bs{self.config.train_batch_size}_ep{self.config.epochs}"
        if self.config.unfreeze_layers is not None:
            file_name += f"_unfreeze{self.config.unfreeze_layers}"

        file_name += ".pt"  # Add file extension for PyTorch model

        # Combine with the base directory
        model_path = base_model_path +file_name
        return model_path
        
  def run_test(self):
      state_dict = torch.load(self.model_path)
      if self.config.model_type.lower() == 'roberta-simple':
          print("Picking Roberta Simple Classifier Model")
          model = dataset.RobertaSimpleClassifier()
      elif self.config.model_type.lower() == 'roberta-gru':
          print("Picking Robert-GRU Model")
          model = dataset.RobertaGRUClassifier()
      else:
          print("Model not supported")
          sys.exit(1)
      model.load_state_dict(state_dict)

      warnings.filterwarnings('ignore')
      #Read the training dataset from the CSV file
      test_raw_dataset = pd.read_csv(constants.TEST_FILE).fillna("none")

      df_test = self.dataset
      df_test = df_test.reset_index(drop=True)

      test_dataset = dataset.RoBERTaDataset(
          review=df_test.review.values, target=df_test.sentiment.values
      )

      test_data_loader = torch.utils.data.DataLoader(
          test_dataset, batch_size=32, num_workers=4
      )
      class_names = ['Negative', 'Neutral', 'Positive']
      engine = Engine(model,self.device,df_test.sentiment.values,self.config.model_type.lower())
      targets, outputs = engine.test_eval_fn(test_data_loader)
      engine.plot_confusion_matrix(targets, outputs, class_names)
      engine.plot_roc_curve(targets, outputs, len(class_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('-m','--model_path', type=str, required=True, help='Path to the model file.')
    parser.add_argument('-t','--model_type', type=str, required=True, help='Model Name')
    args = parser.parse_args()
    test_dataset = pd.read_csv(constants.TEST_FILE)
    preprocessor = dataset.DatasetPreprocessor(test_dataset)
    preprocessed_test_dataset = preprocessor.preprocess_dataset()
    balanced_preprocessed_dataset = pd.read_csv(constants.BALANCED_DATASET_PATH)
    config = Config(3e-5, 32,64, 4, 'roberta-gru')
    config.pretty_print()
    tester = tester = Tester(preprocessed_test_dataset,config)
    Tester.run_test()
