import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc)
from sklearn.model_selection import cross_val_predict


def find_best_threshold(model, X_train, y_train):
    y_fraud_prob = cross_val_predict(
        model, X_train, y_train, cv=3, method="predict_proba"
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_fraud_prob)
    f1_denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        f1_denominator,
        out=np.zeros_like(f1_denominator),
        where=f1_denominator != 0,
    )
    return float(thresholds[f1_scores.argmax()])


def model_evaluation(name, model, X_test, y_test, threshold=0.5):
    y_fraud_prob = model.predict_proba(X_test)[:, 1] 
    y_pred = (y_fraud_prob >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_test, y_fraud_prob)
    
    results = {
		"Model": name,
        "Threshold": threshold,
		"Accuracy":accuracy_score(y_test, y_pred),
		"Precision": precision_score(y_test, y_pred, zero_division=0),
		"Recall":recall_score(y_test, y_pred),
		"F1-Score":f1_score(y_test, y_pred),
		"ROC-AUC": roc_auc_score(y_test, y_fraud_prob),
		"PR-AUC": auc(recall, precision),
	}
    
    print("\n"+name.upper())
    print("-" * 30)
    for test_evaluation, value in results.items():
        if test_evaluation != 'Model':
            print(f"{test_evaluation}: \t{value:.10f}")
    print("\nConfusion Matrix:") 
    print(confusion_matrix(y_test, y_pred))
    print("-"*30)
    return results
