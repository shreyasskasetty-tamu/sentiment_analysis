import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from itertools import cycle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class Engine:
    def __init__(self,model,device,labels,model_type):
        self.model = model
        self.device = device
        self.labels = labels
        self.model_type = model_type

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs.view(-1,3), targets.view(-1))

    def focal_loss_fn(self, outputs, targets, alpha=1, gamma=2):
        focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

        return focal_loss(outputs.view(-1, 3), targets.view(-1))

    def plot_confusion_matrix(self, targets, outputs, class_names):
        # Convert outputs to predicted class indices
        predicted_labels = np.argmax(outputs, axis=1)
        cm = confusion_matrix(targets, predicted_labels)

        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Iterate over the confusion matrix and add labels to each cell
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def plot_roc_curve(self,targets, outputs, n_classes):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        targets_one_hot = np.eye(n_classes)[targets]  # Convert to one-hot encoding
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], np.array(outputs)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        colors = cycle(['blue', 'red', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - multi-class')
        plt.legend(loc="lower right")
        plt.show()

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
        self.model.eval()
        all_targets = []
        all_probabilities = []
        all_predictions = []
        self.model.to(self.device)
        with torch.no_grad():
            for bi, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
                ids = d["ids"].to(self.device, dtype=torch.long)
                mask = d["mask"].to(self.device, dtype=torch.long)
                targets = d["targets"].to(self.device, dtype=torch.long)
                if self.model_type == "bert":
                    token_type_ids = d["token_type_ids"].to(self.device, dtype=torch.long)
                    outputs = self.model(ids,mask,token_type_ids)
                else:  # Assuming RoBERTa or similar
                    outputs = self.model(ids,mask)
                    _, predicted = torch.max(outputs, 1)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_targets.extend(targets.view_as(predicted).cpu().numpy())
                    all_probabilities.extend(probabilities)
                    all_predictions.extend(predicted.cpu().numpy())
        # Calculate F1 Score
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        print(f'F1 Score: {f1}')

        # Calculate and Print Test Accuracy
        test_accuracy = accuracy_score(all_targets, all_predictions)
        print(f'Test Accuracy: {test_accuracy:.2f}')

        # Calculate Per-Class Accuracy
        print('Per-Class Accuracy:')
        unique_labels = set(all_targets)
        for label in unique_labels:
            label_targets = [1 if t == label else 0 for t in all_targets]
            label_predictions = [1 if p == label else 0 for p in all_predictions]
            acc = accuracy_score(label_targets, label_predictions)
            print(f'Accuracy for class {label}: {acc}')

        # One-hot encode the targets
        num_classes = all_probabilities[0].size
        all_targets_one_hot = label_binarize(all_targets, classes=range(num_classes))

        # Calculate AUC for each class
        print('Per-Class AUC Score:')
        auc_scores = {}
        for i in range(num_classes):
            auc_score = roc_auc_score(all_targets_one_hot[:, i], np.array(all_probabilities)[:, i], multi_class='ovr')
            print(f"AUC Scores for Class {i}:", auc_score)

        return all_targets, all_probabilities