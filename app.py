from flask import Flask, render_template, request
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from utils.utils import SentimentAnalysis, SentimentAnalysisModel

app = Flask(__name__)
PORT = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = RobertaForSequenceClassification
tokenizer = RobertaTokenizer
lr = 2e-5 # optimal running rate
optimizer = optim.Adam
criterion = nn.CrossEntropyLoss().to(device)
epochs = 1

SentimentAnalysis(model_name, tokenizer, classifier, num_labels=3, device=device)


@app.route('/', methods=['GET', 'POST'])
def index():
  input_text = ''
  outputs = []
  if request.method == 'POST':
        input_text = request.form['input_text']
        model = SentimentAnalysisModel([input_text])
        outputs = [r for r in model]
        print(outputs[0])
  return render_template('index.html', input_text=input_text, outputs=outputs[0])

if __name__ == '__main__':
    app.run(debug=True)
