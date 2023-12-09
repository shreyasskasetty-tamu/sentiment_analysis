import config
import torch
import flask
import time
from flask import Flask
from flask import request
from model import RoBERTaGRUModel
import functools
import torch.nn as nn


app = Flask(__name__)

MODEL = None
DEVICE = config.DEVICE
PREDICTION_DICT = dict()


def sentence_prediction(sentence):
    tokenizer = config.ROBERTA_TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    # token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask)

    _, predicted = torch.max(outputs, 1)
    print(predicted.cpu().numpy())
    return predicted.cpu().numpy()


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    label = {0:"Negative",1:"Neutral",2:"Positive"}
    prediction = sentence_prediction(sentence)
    print(prediction[0])
    response = {}
    response["response"] = {
        "prediction":label[prediction[0]],
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = RoBERTaGRUModel()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH,map_location='cpu'))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host="0.0.0.0", port="9999")
