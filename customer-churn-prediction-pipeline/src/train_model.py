from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train(X, y, model_name='Random Forest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Unsupported model")

    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    return model, X_test, y_test, cv_score
