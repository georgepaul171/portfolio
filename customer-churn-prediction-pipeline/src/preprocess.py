from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df):
    df = df.dropna(subset=['TotalCharges'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(['customerID'], axis=1)

    for col in df.select_dtypes(include='object'):
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    X = df.drop('Churn', axis=1)
    y = LabelEncoder().fit_transform(df['Churn'])
    return X, y
