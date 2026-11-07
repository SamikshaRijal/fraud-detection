import pandas as pd
from sklearn.linear_model import LogisticRegression

# Add import for models that you are implementing
from preprocessing import get_data
from eval import model_evaluation

X_train, X_test, y_train, y_test = get_data(True)
    
# Logistic regression
logistic_regression= LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
logistic_regression.fit(X_train, y_train)
model_evaluation("Logistic Regression", logistic_regression, X_test, y_test)

# Add your models here

