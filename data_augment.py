import os
import constants
import dataset

import pandas as pd
import nlpaug.augmenter.word as naw

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

def augment_text(text, aug):
    return aug.augment(text)

def augment_data_roberta(data, class_name, target_size=50000, model_path='roberta-base', num_workers=cpu_count()):
    aug = naw.ContextualWordEmbsAug(model_path=model_path, action="insert")
    augmented_texts = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pool_of_texts = data['review'].sample((target_size - len(data)),replace=True) 
        tasks = [executor.submit(augment_text, text, aug) for text in pool_of_texts]
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Augmenting Data", leave=False):
            augmented_text = future.result()
            augmented_texts.append({'review': augmented_text[0], 'sentiment': class_name})
            if len(augmented_texts) >= (target_size - len(data)):
                break

    return pd.DataFrame(augmented_texts)

if __name__ == '__main__':
    if os.path.exists(constants.PREPROCESSED_DATA_PATH):
        print("Loading preprocessed data...")
        preprocessed_dataset = dataset.load_preprocessed_data(constants.PREPROCESSED_DATA_PATH)
        
    else:
        print("Preprocessing data...")
        raw_data_df = pd.read_csv(constants.TRAINING_FILE)
        proccessor = dataset.DatasetPreprocessor(raw_data_df)
        preprocessed_dataset = proccessor.preprocess_dataset()
        dataset.save_preprocessed_data(preprocessed_dataset, constants.PREPROCESSED_DATA_PATH)

    preprocessed_dataset = preprocessed_dataset.reset_index(drop=True)
    # Augment 'Negative' class
    negative_reviews = preprocessed_dataset[preprocessed_dataset['sentiment'] == 0]
    augmented_negative_reviews = augment_data_roberta(negative_reviews, 0, target_size=50000)
    # Append augmented data to the original negative reviews
    extended_negative_reviews = pd.concat([negative_reviews, augmented_negative_reviews])
    # Augment 'Neutral' class
    neutral_reviews = preprocessed_dataset[preprocessed_dataset['sentiment'] == 1]
    augmented_neutral_reviews = augment_data_roberta(neutral_reviews, 1, target_size=50000)
   
    # Append augmented data to the original neutral reviews
    extended_neutral_reviews = pd.concat([neutral_reviews, augmented_neutral_reviews])

    # Downsample 'Positive' class
    downsampled_positive_reviews = preprocessed_dataset[preprocessed_dataset['sentiment'] == 2].sample(n=50000, random_state=constants.RANDOM_STATE)

    # Combine all three classes
    balanced_data = pd.concat([downsampled_positive_reviews, extended_negative_reviews, extended_neutral_reviews])
    dataset.save_preprocessed_data(balanced_data,constants.BALANCED_DATASET_PATH)