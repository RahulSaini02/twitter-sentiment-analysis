import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer

class SentimentAnalysis(nn.Module):
  def __init__(self, model_name, tokenizer, classifier, num_labels, device):
    super(SentimentAnalysis, self).__init__()
    self.model_name = model_name
    self.tokenizer = tokenizer.from_pretrained(self.model_name)
    self.classifier = classifier
    self.num_labels = num_labels
    self.device = device

    self.model = self.classifier.from_pretrained(model_name, num_labels=self.num_labels).to(self.device)

  def forward(self, input_ids, attention_mask):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs

  def fit(self, train_loader, criterion, opt_fn, lr, epochs):
    self.model.train()
    losses = []
    optimizer = opt_fn(self.model.parameters(), lr)
    for epoch in range(epochs):
      running_loss = 0.0
      for batch in train_loader:
          input_ids = batch["input_ids"].to(self.device)
          attention_mask = batch["attention_mask"].to(self.device)
          labels = batch["label"].to(self.device)

          optimizer.zero_grad()

          outputs = self.forward(input_ids, attention_mask)
          loss = criterion(outputs.logits, labels)

          loss.backward()
          optimizer.step()

          running_loss += loss.item()

      losses.append(running_loss)

    avg_loss = sum(losses) / len(train_loader)
    return avg_loss

  def predict(self, test_loader, criterion):
    self.model.eval()
    total_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.forward(input_ids, attention_mask)
            _, preds = torch.max(outputs.logits, 1)

            correct_predictions += torch.sum(preds == labels).item()

            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
    accuracy = correct_predictions / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss

class SentimentAnalysisModel:
    def __init__(self, tweets, model_path='utils/sentimentmodel.h5', max_length=512):
        self.tweets = tweets
        self.tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model on CPU
        self.max_length = max_length

    def tokenized_value(self, tweet):
        text = str(tweet)
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(1),
            "attention_mask": encoding["attention_mask"].squeeze(1),
        }

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        tweet_tensor = self.tokenized_value(tweet)
        outputs = self.model(tweet_tensor["input_ids"], tweet_tensor["attention_mask"])
        probabilities = F.softmax(outputs.logits, dim=1)
        scores = probabilities.tolist()[0]
        sentiment_labels = ["negative", "neutral", "positive"]
        results = []
        for x in range(len(scores)):  
            result = {}   
            result['label'] = sentiment_labels[x]
            result['score'] = "{:.3f}".format(scores[x])
            results.append(result)

        return results


