import transformers
import torch

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
DEVICE = "cpu"
RANDOM_STATE = 42
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
BASE_MODEL_PATH = "models/"
TRAINING_FILE = "dataset/yelp_review_train.csv"
TEST_FILE = "dataset/yelp_review_test.csv"
PREPROCESSED_DATA_PATH = "dataset/preprocessed_yelp_review.csv"
BALANCED_DATASET_PATH = "dataset/balanced_yelp_review_train.csv"
BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
ROBERTA_TOKENIZER =  transformers.RobertaTokenizer.from_pretrained('roberta-base')
STOP_WORDS_DOWNLOAD_PATH = "dataset/"
LEARNING_RATE = 3e-5
FINE_TUNING_LEARNING_RATE = 3e-6