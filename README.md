# Fake News Detection using NLP and Machine Learning

A Natural Language Processing (NLP) and Machine Learning project that classifies news articles as **FAKE** or **REAL** based on their textual content.

The project covers the complete NLP machine learning workflow, including data cleaning, text preprocessing, feature extraction, model comparison, hyperparameter tuning, and final model selection.

---

## 📌 Project Overview

The objective of this project is to build a machine learning model capable of classifying news articles as either **FAKE** or **REAL** based on their textual content.

The complete workflow includes:

- Data cleaning
- Text preprocessing
- Tokenization
- Stopword removal
- Lemmatization
- Feature extraction using Bag of Words
- Feature extraction using TF-IDF
- Model training and comparison
- Hyperparameter tuning
- Final model selection
- Model serialization
- Prediction pipeline testing

---

## 📂 Dataset

The dataset contains news articles labeled as:

- **FAKE**
- **REAL**

The dataset contains the following main columns:

- `title`
- `text`
- `label`

The `title` and `text` columns were combined into a single feature called `context`, which was used for text classification.

---

## 🧹 Data Cleaning

The dataset was checked for:

- Duplicate records
- Missing values
- Empty text entries
- Class distribution

Duplicate records and rows with empty article text were removed before training the models.

---

## 🔤 Text Preprocessing

The following preprocessing steps were applied to the text:

1. Converted text to lowercase
2. Removed URLs
3. Removed special characters and punctuation
4. Tokenized the text
5. Removed English stopwords
6. Applied lemmatization
7. Combined processed tokens back into text

### Preprocessing Pipeline

```text
Raw News Article
        ↓
Lowercase
        ↓
Remove URLs
        ↓
Remove Special Characters
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
Cleaned Text
```

---

## 🔢 Feature Extraction

Two feature extraction techniques were used to convert text into numerical features.

### Bag of Words (BoW)

Bag of Words represents text based on word frequency.

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) assigns importance to words based on their frequency within a document and across the dataset.

---

## 🤖 Models Used

The following machine learning models were trained and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Classifier (SVC)
- Linear Support Vector Classifier (LinearSVC)

Both **Bag of Words** and **TF-IDF** feature representations were tested.

---

## ⚙️ Hyperparameter Tuning

The `C` parameter was tuned for both SVC and LinearSVC models.

The following values were tested:

```text
0.01, 0.1, 1, 2, 5, 10, 20
```

The best-performing configuration was selected based on test performance.

---

## 🏆 Final Model

The final model selected was:

**Support Vector Classifier (SVC) with TF-IDF features**

Best configuration:

```text
C = 2
```

### Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 94.42% |
| Training Accuracy | 100% |

### Confusion Matrix

```text
[[607  16]
 [ 54 577]]
```

The model achieved strong performance in distinguishing between FAKE and REAL news articles.

---

## 🔮 Prediction Pipeline

The complete prediction workflow is:

```text
Raw News Text
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Trained SVC Model
      ↓
FAKE / REAL Prediction
```

A reusable preprocessing function ensures that new input text goes through the same preprocessing steps used during model training.

---

## 💾 Model Saving

The trained model and TF-IDF vectorizer were saved using `joblib`.

```python
import joblib

joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(final_model, "final_model.pkl")
```

The saved files can later be loaded for prediction or deployment.

```python
loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")
loaded_model = joblib.load("final_model.pkl")
```

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── NLP.ipynb
├── fake_or_real_news.csv
├── final_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 🚀 Future Improvements

Possible improvements for this project include:

- Building a web interface using Streamlit
- Creating an API using FastAPI
- Deploying the model
- Using n-grams for feature extraction
- Performing cross-validation
- Experimenting with deep learning models such as LSTM
- Using transformer-based models such as BERT

---

## 👤 Author

**Khush Pawar**

Aspiring Data Scientist | Machine Learning Enthusiast

---

> **Note:** This model is trained on a specific labeled dataset and performs text classification based on patterns learned from that data. It should not be considered a real-world fact-checking system.