# Sentiment Analysis with Transformers

![Transformers](https://img.shields.io/badge/NLP-Transformers-blue)

A transformer-based sentiment analysis system trained on Twitter data using Hugging Face Transformers and deployed via Flask.

---

## 📌 Overview
This project utilizes **RoBERTa**, a transformer model, to classify tweet sentiments as **positive**, **neutral**, or **negative**. The model is trained using the `tweet_eval` dataset and deployed with a simple web interface using Flask.

---

## 📂 Dataset
- **Name**: [tweet_eval](https://huggingface.co/datasets/tweet_eval)
- **Description**: Pre-labeled tweet dataset for sentiment classification.
- **Classes**: Positive, Neutral, Negative

---

## ⚙️ Features
- Pretrained transformer-based sentiment classification using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Model fine-tuned and evaluated on tweet_eval dataset
- Web application with Flask for live predictions
- Frontend with interactive input and styled output

---

## 🔧 Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/sentiment-analysis-transformers.git
cd sentiment-analysis-transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Flask App
```bash
# Run the application
python app.py
```
- Visit `http://localhost:5000/` to interact with the web app

---

## 🧠 Model Architecture
- **Tokenizer**: RobertaTokenizer
- **Model**: RobertaForSequenceClassification
- **Trained On**: tweet_eval (Sentiment Subset)
- **Accuracy**: 71.88%

---

## 🧪 Evaluation Metrics
| Metric        | Value   |
|---------------|---------|
| Training Loss | 1.31    |
| Test Loss     | 0.74    |
| Accuracy      | 71.88%  |

---

## 📁 File Structure
```
/app
├── static/                # Static assets (images, CSS)
├── templates/             # HTML templates
├── utils/                 # Model wrapper and custom classes
├── app.py                 # Flask backend
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

---

## 🌐 Deployment
The model is deployed with Flask and supports real-time sentiment classification via web input.

**Sample Output:**
```json
[
  { "label": "negative", "score": 0.723 },
  { "label": "neutral", "score": 0.228 },
  { "label": "positive", "score": 0.047 }
]
```

---

## 🛠 Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Flask
- HTML/CSS

---
