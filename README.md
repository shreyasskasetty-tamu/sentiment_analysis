# Yelp Restaurant Review Rating Semtiment Analysis using Transformer Models
Sentiment Analysis of Yelp Review Rating Dataset Using Transformer Models

# Introduction

This project focuses on sentiment analysis of Yelp restaurant reviews using transformer models. Leveraging the advanced capabilities of models like BERT and RoBERTa, the project aims to accurately classify reviews into different sentiment categories. This approach allows for a nuanced understanding of customer feedback, vital for businesses and analytics.

# Model Details

## Architecture
Our project employs the BERT (Bidirectional Encoder Representations from Transformers) and RoBERTa (Robustly Optimized BERT Pretraining Approach) models. These transformer models are renowned for their effectiveness in natural language processing tasks. 

### BERT Model
- **Base**: We use the 'bert-base-uncased' variant of BERT as our foundational model.
- **Custom Layer**: A dense layer with 512 neurons is added to the base model, followed by layer normalization and a dropout rate of 0.3.
- **Output Layer**: The model concludes with a three-unit output layer, aligning with our classification categories.

### RoBERTa-GRU Hybrid Model
- **Integration of RoBERTa and GRU**: This model synergizes the RoBERTa transformer with Gated Recurrent Units (GRU). RoBERTa handles the embedding of texts, while GRU layers manage long-range dependencies and mitigate the vanishing gradient issue.
- **Data Augmentation for Imbalance**: To address imbalanced datasets, we implement data augmentation with word embeddings, focusing on oversampling minority classes.

## Training Adaptations
- **Parameter Freezing**: For efficient training, the original pretrained model parameters are frozen, utilizing them primarily for feature extraction.

# Usage

To use this project for sentiment analysis of Yelp restaurant reviews, follow these steps:

## Prerequisites
- Ensure you have Python installed on your system.
- Install necessary libraries such as PyTorch, Transformers, and Pandas. These can be installed via pip or conda.

## Data Preparation
- The model requires preprocessed Yelp restaurant review data. Follow the preprocessing steps outlined in the `data_augment.py` and `dataset.py` scripts.

## Running the Models
1. **BERT Model**:
   - To train the BERT model, execute the `train.py` script with the necessary arguments.
   - Use the `test.py` script to evaluate the model on the test dataset.

2. **RoBERTa-GRU Model**:
   - Run the `train.py` script with specified flags for the RoBERTa-GRU model.
   - Test the model using the `test.py` script, which will output performance metrics.

For more detailed instructions, please refer to the documentation and comments within each script.
