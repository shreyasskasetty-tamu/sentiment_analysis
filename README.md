# Yelp Restaurant Review Rating Sentiment Analysis using Transformer Models
Sentiment Analysis of Yelp Review Rating Dataset Using Transformer Models

## Getting Started

To begin, clone this repository to your local machine or download it as a zip file.
Repository link: https://github.com/shreyasskasetty-tamu/sentiment_analysis

## Contents

This repository includes the following key components:

1. **Jupyter Notebook:**
   - `SentimentAnalysis.ipynb`: This notebook contains the code and explanations for the sentiment analysis model. It is located in the `notebooks` folder.

2. **Documentation:**
   - `ML_Sentiment_Analysis_Project.pdf`: A detailed report of the project, available in the `documentation` folder.
   - `Sentiment Analysis CSCE 633.pptx`: PowerPoint presentation that provides an overview of the project, also located in the `documentation` folder.
   - `ML Sentiment Analysis Project.zip`: Additional documentation resources packed in a zip file, found in the `documentation` folder.

3. **Source Code:**
   - Files like `app.py`, `train.py`, `test.py`, and others in the main directory for implementing and testing the model.

4. **Video Presentation:**
   - Video presentation can be found in the `video` directory. It needs to be downloaded inorder to be viewed.
   - The video can also be accessed from the drive link: https://drive.google.com/file/d/1pebGJGJstYnneGskbhgFpcCtzC-JiF9X/view?usp=drive_link
   - 

5. **Data:**
   - The `dataset` directory contains the data used for training and testing the model.

6. **Best Model Weights:**
   - Best Model weights can be found in the drive link: https://drive.google.com/file/d/1HG1xaWLq7BCJrVdC40yeqnlvdLzJBy_b/view?usp=drive_link. This model can be used in the test script to get the performance of the model. The test scripts also outputs graph plots for visualization.

## How to Use

### Jupyter Notebook
- Navigate to the `notebooks` folder.
- Open `SentimentAnalysis.ipynb` in a Jupyter environment to view and run the code.

### Documentation
- Reports and presentations can be found under the `documentation` folder.
- You can open the PDF and PPT files with any compatible reader.

# Usage

To use this project for sentiment analysis of Yelp restaurant reviews, follow these steps:

## Prerequisites
- Ensure you have Python installed on your system.
- Install necessary libraries such as PyTorch, Transformers, and Pandas. These can be installed via pip or conda.

## Data Preparation
- The model requires preprocessed Yelp restaurant review data. Follow the preprocessing steps outlined in the `data_augment.py` and `dataset.py` scripts.
- The repository already has preprocessed dataset in the dataset directory. Path for different dataset can be assigned in the constants file for the constant `BALANCED_DATASET_PATH`

## Training and Testing the Model

This project includes scripts for training and testing the sentiment analysis model. Follow these instructions to run these processes.

### Training the Model

To train the model, use the following command:

```bash
python train.py -t <model_type>
```

To test the model, use the following command:

```bash
python test.py -m <model_path> -t <model_type>
```

# Introduction

This project focuses on sentiment analysis of Yelp restaurant reviews using transformer models. Leveraging the advanced capabilities of pretrained models like RoBERTa, the project aims to accurately classify reviews into different sentiment categories. This approach allows for a nuanced understanding of customer feedback, vital for businesses and analytics.


## Methodology

The core of my sentiment analysis model is the RoBERTa-GRU hybrid architecture. This innovative approach combines the strengths of RoBERTa's deep transformer-based encoding capabilities with the dynamic sequence modeling proficiency of Gated Recurrent Units (GRUs). The RoBERTa model, a robustly optimized BERT pretraining approach, serves as a poIrful encoder that captures the context within the text at a granular level. Following this, the GRU layers work to analyze the sequence of embeddings, considering the temporal dependencies that are crucial for understanding the sentiments expressed over time in reviews.

### Model Training Parameters

The model's training process is carefully calibrated using the categorical cross-entropy loss function, which measures the performance of the classification model whose output is a probability value betIen 0 and 1. The loss increases as the predicted probability diverges from the actual label, making it an ideal choice for the multi-class sentiment classification task. I optimize the training using the AdamW optimizer, an extension of the traditional Adam algorithm, known for its effective handling of sparse gradients and adaptive learning rate capabilities. AdamW introduces Iight decay to the standard Adam optimizer, which helps in regularizing and preventing overfitting.

## Dataset and Preprocessing

The dataset comprises Yelp reviews, a rich source of user-generated content expressing a wide range of sentiments. The preprocessing pipeline is designed to clean and normalize this data, enhancing the model's ability to learn from it effectively. Text normalization ensures consistency in the dataset, transforming all text to loIr case. Stop words, which are frequently occurring words that carry minimal useful information for analysis, are removed. Punctuation is also stripped away, leaving behind a dataset focused on the core content. In addition to these steps, data augmentation techniques are employed to address class imbalances. By generating synthetic samples, I enrich the dataset, ensuring that the model can learn to identify each sentiment class accurately.

## Hyperparameter Tuning

Tuning the model's hyperparameters is a critical step in achieving peak performance. I adopted a systematic grid search strategy to iterate through a range of learning rates, training batch sizes, and epoch numbers to find the most effective combination. This extensive search was vital for understanding how each hyperparameter affects model performance and ensured that I could fine-tune the model to respond optimally to the nuances of the specific dataset. The optimal parameters Ire determined based on their impact on the validation set's performance, aiming for the highest F1 score—a balanced measure considering both precision and recall—and overall accuracy.

### Hyperparameter Tuning Analysis

The table below presents a comprehensive overview of hyperparameter tuning results from a grid search performed on a sentiment analysis task. The configurations explore combinations of different learning rates, batch sizes, and epoch counts, along with a distinction between two model architectures: `roberta-simple` and `roberta-gru`. Additionally, some configurations experiment with unfreezing layers during training, which can lead to significant performance differences.

| No. | Model Type                          | Learning Rate | Train BS | Val BS | Epochs | Unfreeze Layers | F1 Score | Test Acc. |
|-----|------------------------------------|---------------|----------|--------|--------|-----------------|----------|-----------|
| 1   | roberta-simple                     | 1e-05         | 16       | 32     | 3      | None            | 0.72864  | 0.71      |
| 2   | roberta-simple                     | 2e-05         | 32       | 64     | 3      | None            | 0.73189  | 0.71      |
| 3   | roberta-simple                     | 1e-05         | 16       | 32     | 6      | None            | 0.76558  | 0.75      |
| 4   | roberta-gru                        | 1e-05         | 16       | 32     | 3      | None            | 0.78996  | 0.78      |
| 5   | roberta-gru                        | 2e-05         | 32       | 64     | 3      | None            | 0.78818  | 0.77      |
| 6   | roberta-gru                        | 1e-05         | 16       | 32     | 6      | None            | 0.79599  | 0.78      |
| 7   | roberta-simple                     | 3e-05         | 32       | 64     | 4      | None            | 0.76120  | 0.74      |
| 8   | roberta-gru                        | 3e-05         | 32       | 64     | 4      | None            | 0.79310  | 0.77      |
| 9   | roberta-simple                     | 1e-05         | 64       | 64     | 5      | None            | 0.71075  | 0.69      |
| 10  | roberta-gru                        | 1e-05         | 64       | 64     | 5      | None            | 0.78408  | 0.77      |
| 11  | roberta-gru                        | 1e-05         | 32       | 64     | 2+8     | 6               | 0.86048  | 0.85      |
| 12  | roberta-gru                        | 1e-05         | 32       | 64     | 2+8   | 12              | 0.87406  | 0.87      |
| 13  | roberta-gru                        | 1e-05         | 32       | 64     | 2+8    | 6               | 0.86048  | 0.85      |
| 14  | roberta-gru                        | 1e-05         | 32       | 64     | 2+8    | 12              | 0.87406  | 0.87      |

From the results, it is evident that both model types benefit from a fine-tuned learning rate of `1e-05`, which consistently yields higher F1 scores across various configurations. The `roberta-gru` model shows a marked improvement in performance when the number of epochs is increased and layers are unfrozen, with the F1 score reaching as high as `0.87406` and test accuracy peaking at `0.87`.

The batch size appears to have a less consistent impact on performance, but larger batch sizes combined with a higher number of epochs and layer unfreezing tend to produce better results. For instance, configuration 12, which uses the `roberta-gru` model with 12 layers unfrozen and an extended training period (`2+10` epochs), achieves the highest F1 score and test accuracy.

Given the data, the optimal hyperparameters for this specific sentiment analysis task appear to be:
- **Model Type**: `roberta-gru`
- **Learning Rate**: `1e-05`
- **Training Batch Size**: `32`
- **Validation Batch Size**: `64`
- **Epochs**: `10` (which could be interpreted as `2+8` based on the given configurations)
- **Unfreeze Layers**: `12`

The decision to unfreeze 12 layers indicates that allowing more flexibility in the pre-trained model's parameters significantly contributes to the model's ability to learn from the domain-specific data. However, it's crucial to note that while these parameters yield the best results in the grid search, they should be validated on an independent test set to ensure that the model has not overfitted to the validation set used during tuning.

Overall, the `roberta-gru` architecture with carefully chosen hyperparameters and a strategy that involves unfreezing layers offers a robust approach for improving sentiment analysis performance on the given dataset.

## Experiment and Result Analysis

Experiments were conducted with and without dataset augmentation and balancing, with a separate approach using gradual unfreezing of pretrained RoBERTa layers.

### With Dataset Augmentation and Balancing

- F1 Score: 0.885
- Test Accuracy: 0.88
- AUC Scores: 0.985 (Negative), 0.897 (Neutral), 0.975 (Positive)

### Without Dataset Augmentation and Balancing

- F1 Score: 0.883
- Test Accuracy: 0.89
- AUC Scores: 0.986 (Negative), 0.902 (Neutral), 0.977 (Positive)

### Gradual Unfreezing of Pretrained RoBERTa Layers

- F1 Score: 0.827
- Test Accuracy: 0.81
- AUC Scores: 0.967 (Negative), 0.835 (Neutral), 0.952 (Positive)

## Conclusion

The RoBERTa-GRU hybrid model demonstrates strong performance in sentiment analysis of Yelp reviews, with optimal hyperparameters enhancing the learning efficiency and overall model performance.
