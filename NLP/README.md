# SMS Spam Classifier

## Project Overview

This project implements a simple Natural Language Processing (NLP) pipeline to classify SMS messages as either `ham` (legitimate) or `spam` (unsolicited). The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository and uses TF-IDF vectorisation combined with a logistic regression classifier to perform the classification.

## Dataset

- **Name:** SMS Spam Collection
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Description:** This dataset is a collection of SMS messages that have been labeled as either `ham` (legitimate) or `spam`. Each line in the dataset consists of a label and a message separated by a tab.

**Note:**  
Make sure to download the dataset file (commonly named `SMSSpamCollection`) and place it in the project directory alongside the Python script.

## How It Works

1. **Data Loading:**  
   The script loads the dataset using `pandas.read_csv`, specifying a tab (`\t`) as the separator. It also assigns column names (`label` and `message`) since the file does not include a header row.

2. **Data Splitting:**  
   The data is split into training and testing sets using an 80/20 ratio. The splitting is stratified to maintain the same proportion of `ham` and `spam` messages in both sets.

3. **Text Vectorisation:**  
   The messages are converted to numerical features using a TF-IDF vectorizer, which transforms the text data into term-frequency inverse document frequency (TF-IDF) features. This helps in capturing the importance of words in the messages.

4. **Model Training:**  
   A logistic regression classifier is trained on the TF-IDF features from the training data.

5. **Evaluation:**  
   The trained model is then used to predict labels on the test data. A classification report is generated, summarizing key metrics like precision, recall, F1-score, and overall accuracy.


### Setting Up Your Environment

1. **Clone or download the project repository.**

2. **Create and activate a virtual environment:**

   ```bash
   cd /path/to/your/project-directory/NLP
   python -m venv venv
   source venv/bin/activate   # On Windows, use: venv\Scripts\activate

3.	**Install required packages:**
   pip install pandas scikit-learn

4. **Place the dataset in project directory**

5. **Run the script:**
   python spam_classifier.py


## Expected Output

### Dataset Preview:
The first 5 rows of the dataset are printed to confirm it loaded correctly.

### Dataset Summary:
A count of `ham` and `spam` messages is shown, displaying the class distribution.

### Classification Report:
Example output:

### Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| ham           | 0.96      | 1.00   | 0.98     | 966     |
| spam          | 1.00      | 0.76   | 0.86     | 149     |
| **accuracy**  |           |        | 0.97     | 1115    |
| **macro avg** | 0.98      | 0.88   | 0.92     | 1115    |
| **weighted avg** | 0.97   | 0.97   | 0.97     | 1115    |


- **Precision:** How many predicted positives are truly positive.
- **Recall:** How many actual positives are correctly predicted.
- **F1-score:** Harmonic mean of precision and recall.
- **Accuracy:** Overall correctness of predictions.

## Next Steps

- **Enhanced Preprocessing:**  
  Add more text cleaning (e.g., punctuation removal, lowercasing, lemmatization).

- **Try Different Models:**  
  Try Naive Bayes, SVM, or deep learning models for comparison.

- **Deploy:**  
  Package the model into a Flask or Streamlit app for real-time SMS classification.

---

## Conclusion

This project demonstrates a complete NLP pipeline: from data loading to preprocessing, vectorization, model training, and evaluation. It’s a solid portfolio project to showcase fundamental NLP and ML skills.
