import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Load the dataset
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=["label", "message"])

# Inspect the dataframe
print("Dataset preview:")
print(df.head())
print("\nDataset summary:")
print(df['label'].value_counts())


# Separate features and labels.
X = df['message']
y = df['label']

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert the text data into numerical features using TF-IDF vectorization.
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a logistic regression classifier.
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

# Make predictions on the test set.
y_pred = lr.predict(X_test_tfidf)

# Evaluate the classifier's performance.
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))