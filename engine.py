import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from collections import Counter

class Engine:
    def __init__(self,model,device,labels,model_type):
        self.model = model 
        self.device = device
        self.labels = labels
        self.model_type = model_type
        # self.weights = self.compute_class_weights()

    # def compute_class_weights(self):
    #     """
    #     Computes class weights as a PyTorch tensor based on class frequencies in the dataset.

    #     :param labels: Array-like, the class labels in the dataset.
    #     :return: A PyTorch tensor containing weights for each class.
    #     """
    #     class_counts = Counter(self.labels)
    #     total_samples = len(self.labels)
    #     num_classes = len(class_counts)
    #     # Calculating weight for each class as inverse of its frequency
    #     weights = [total_samples / (num_classes * class_counts[class_label]) for class_label in class_counts]

        # return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def loss_fn(self, outputs, targets, is_weighted = False):
        # if is_weighted:
        #     return nn.CrossEntropyLoss(weight=self.weights)(outputs.view(-1,3), targets.view(-1))
        # else:
        return nn.CrossEntropyLoss()(outputs.view(-1,3), targets.view(-1))

    def train_fn(self, data_loader_tqdm, optimizer, scheduler):
        self.model.train()
        loss = None

        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for bi, d in data_loader_tqdm:
            ids = d["ids"].to(self.device, dtype=torch.long)
            mask = d["mask"].to(self.device, dtype=torch.long)
            targets = d["targets"].to(self.device, dtype=torch.long)

            optimizer.zero_grad()

            if self.model_type == "bert":
                token_type_ids = d["token_type_ids"].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids)
            else:  # Assuming RoBERTa or similar
                outputs = self.model(ids, mask)

            loss = self.loss_fn(outputs, targets)
            
            
            total_loss += loss.item()
            num_batches += 1
            preds = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
            total_accuracy += accuracy
            # Calculate cumulative average
            cumulative_avg_loss = total_loss / num_batches
            cumulative_avg_accuracy = total_accuracy / num_batches
            # Update tqdm description with cumulative average loss and accuracy
            data_loader_tqdm.set_description(f'Loss: {cumulative_avg_loss:.4f},Accuracy: {cumulative_avg_accuracy:.4f}')

            loss.backward()
            optimizer.step()
            scheduler.step()
        return total_loss / len(data_loader_tqdm), total_accuracy / len(data_loader_tqdm)


    def eval_fn(self,data_loader_tqdm):
        self.model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for bi, d in data_loader_tqdm:
                ids = d["ids"].to(self.device, dtype=torch.long)
                mask = d["mask"].to(self.device, dtype=torch.long)
                targets = d["targets"].to(self.device, dtype=torch.long)

                if self.model_type == "bert":
                    token_type_ids = d["token_type_ids"].to(self.device, dtype=torch.long)
                    outputs = self.model(ids, mask, token_type_ids)
                else:  # Assuming RoBERTa or similar
                    outputs = self.model(ids,mask)
                
                # Convert model outputs to probabilities and then to class indices
                probs = torch.softmax(outputs, dim=1)
                fin_outputs.extend(probs.cpu().detach().numpy().tolist())
                # Add the true labels
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    def test_eval_fn(self, test_data_loader):
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.device)
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            with tqdm(enumerate(test_data_loader), total=len(test_data_loader)) as test_data_loader_tqdm:
                for bi, d in test_data_loader_tqdm:
                    ids = d["ids"].to(self.device, dtype=torch.long)
                    mask = d["mask"].to(self.device, dtype=torch.long)
                    targets = d["targets"].to(self.device, dtype=torch.long)
                    if self.model_type == "bert":
                        token_type_ids = d["token_type_ids"].to(self.device, dtype=torch.long)
                        outputs = self.model(ids,mask,token_type_ids)
                    else:  # Assuming RoBERTa or similar
                        outputs = self.model(ids,mask)
                        _, predicted = torch.max(outputs, 1)

                        all_targets.extend(targets.view_as(predicted).cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())

        # Calculate F1 Score
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        print(f'F1 Score: {f1}')

        # Calculate and Print Test Accuracy
        test_accuracy = accuracy_score(all_targets, all_predictions)
        print(f'Test Accuracy: {test_accuracy:.2f}')

        # Calculate Per-Class Accuracy
        unique_labels = set(all_targets)
        for label in unique_labels:
            label_targets = [1 if t == label else 0 for t in all_targets]
            label_predictions = [1 if p == label else 0 for p in all_predictions]
            acc = accuracy_score(label_targets, label_predictions)
            print(f'Accuracy for class {label}: {acc}')

