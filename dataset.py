import os
import config
import torch
import nltk
import regex as re
import pandas as pd
import nlpaug.augmenter.word as naw
import warnings

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from tqdm import tqdm

# TODO: Remove these two functions
def save_preprocessed_data(data, filepath):
    data.to_csv(filepath, index=False)

def load_preprocessed_data(filepath):
    return pd.read_csv(filepath)

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.BERT_TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        target = self.target[item]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.long),
        }
        
class RoBERTaDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.ROBERTA_TOKENIZER  # Ensure this is a RoBERTa tokenizer
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # RoBERTa doesn't use token type IDs, so we don't need to include them here

        target = self.target[item]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.long),
        }

class DatasetPreprocessor:
    def __init__(self, dataset_df,text_col="text",label_col="stars"):
        self.dataset_df = dataset_df
        self.text_col = text_col
        self.label_col = label_col
        tqdm.pandas()
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from the given text.

        Args:
        text (str): The text to be processed.

        Returns:
        str: The text with punctuation removed.
        """
        return re.sub(r'[^\w\s]', '', text)

    def convert_to_lowercase(self, text: str) -> str:
        """
        Convert all characters in the given text to lowercase.

        Args:
        text (str): The text to be processed.

        Returns:
        str: The text in lowercase.
        """
        return text.lower()
    
    def download_stopwords(self, download_dir):
        # Define the path where NLTK will store the stopwords
        nltk.data.path.append(download_dir)

        # Check if the stopwords are already downloaded
        if not os.path.exists(os.path.join(download_dir, "corpora/stopwords")):
            # Download stopwords
            nltk.download("stopwords", download_dir=download_dir)
        else:
            print("Stopwords already downloaded.")

    def remove_stopwords(self, text: str, stopwords: set) -> str:
        """
        Remove stopwords from the given text.

        Args:
        text (str): The text to be processed.
        stopwords (set): A set of stopwords to remove from the text.

        Returns:
        str: The text with stopwords removed.
        """
        return ' '.join([word for word in text.split() if word not in stopwords])
    
    def categorize_stars(self, stars: float) -> str:
        """
        Categorize the star rating into Positive, Negative, and Neutral.

        Args:
        stars (float): The star rating.

        Returns:
        str: The category of the rating (Positive, Negative, Neutral).
        """
        if stars > 3:
            return "Positive"
        elif stars == 3:
            return "Neutral"
        else:
            return "Negative"

    def augment_data_roberta(self,data, class_name, target_size=20000, model_path='roberta-base'):
        aug = naw.ContextualWordEmbsAug(model_path=model_path, action="insert")
        augmented_texts = []

        while len(augmented_texts) + len(data) < target_size:
            for text in tqdm(data['review'], desc="Augmenting Data"):
                augmented_text = aug.augment(text)
                augmented_texts.append({'review': augmented_text, 'sentiment': class_name})
                if len(augmented_texts) + len(data) >= target_size:
                    break

        return pd.DataFrame(augmented_texts)

    def preprocess_dataset(self):

        print("Preprocessing Dataset...")
        print("1. Removing punctuations")
        #Apply remove_punctuation function to the text column
        self.dataset_df['text_no_punctuation'] = self.dataset_df['text'].progress_apply(self.remove_punctuation)
        
        print("2. Converting text to lower case")
        # Apply the convert_to_lowercase function to the text_no_punctuation column
        self.dataset_df['text_lowercase'] = self.dataset_df['text_no_punctuation'].progress_apply(self.convert_to_lowercase)

        # Downlaod stop words in English Language
        print("3. Removing Stop Words")

        self.download_stopwords(config.STOP_WORDS_DOWNLOAD_PATH)
        
        english_stopwords = stopwords.words('english')
        
        self.dataset_df['text_no_stopwords'] = self.dataset_df['text_lowercase'].progress_apply(lambda x: self.remove_stopwords(x, english_stopwords))
        
        print("4. Converting to stars to sentiment")
        # Apply the categorize_stars function to the stars column
        self.dataset_df['sentiment_category'] = self.dataset_df['stars'].progress_apply(self.categorize_stars)

        # Creating a dataframe with only the necessary columns for analysis
        final_preprocessed_data = self.dataset_df[['text_no_stopwords', 'sentiment_category']]

        # Renaming the columns for clarity
        final_preprocessed_data.columns = ['review', 'sentiment']
        
        # Data Augmentation
        # # Assuming train_data is your DataFrame and you want to augment the 'Negative' class
        # negative_reviews = final_preprocessed_data[final_preprocessed_data['sentiment'] == 'Negative']
        # augmented_negative_reviews = self.augment_data_roberta(negative_reviews, 'Negative', target_size=20000)
        
        # # Data Augmentation
        # # Assuming train_data is your DataFrame and you want to augment the 'Negative' class
        # neutral_reviews = final_preprocessed_data[final_preprocessed_data['sentiment'] == 'Neutral']
        # augmented_neutral_reviews = self.augment_data_roberta(neutral_reviews, 'Neutral', target_size=20000)
        # downsampled_positive_reviews = final_preprocessed_data[final_preprocessed_data['sentiment'] == 'Positive'].sample(n=20000, random_state=config.RANDOM_STATE)
        # balanced_data = pd.concat([downsampled_positive_reviews, augmented_negative_reviews, augmented_neutral_reviews])
        # #Encoding the labels to numerical values
        
        # Initializing the label encoder
        label_encoder = LabelEncoder()
        
        print("5. Enconding sentiments to 0, 1, 2 (Positive, Negative, Neutral)")
        # Applying label encoding to the sentiment column
        final_preprocessed_data['sentiment'] = label_encoder.fit_transform(final_preprocessed_data['sentiment'])

        # Display the mapping of categories to integers and the first few rows of the dataset
        category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Finalizing the preprocessed DataFrame with only the necessary columns for model training
        preprocessed_data = final_preprocessed_data[['review', 'sentiment']]
        print("Preprocessing done!")
        return preprocessed_data

#NOTE: Comment the main function which is being used for debugging
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Check if preprocessed data exists
    if os.path.exists(config.PREPROCESSED_DATA_PATH):
        print("Loading preprocessed data...")
        dfx = load_preprocessed_data(config.PREPROCESSED_DATA_PATH)
    else:
        print("Preprocessing data...")
        raw_data_df = pd.read_csv(config.TRAINING_FILE)
        proccessor = DatasetPreprocessor(raw_data_df)
        dfx = proccessor.preprocess_dataset()
        save_preprocessed_data(dfx, config.PREPROCESSED_DATA_PATH)
