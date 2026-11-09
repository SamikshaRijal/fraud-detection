from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc)

def model_evaluation(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_fraud_prob = model.predict_proba(X_test)[:, 1] 
    precision, recall, _ = precision_recall_curve(y_test, y_fraud_prob)
    
    results = {
		"Model": name,
		"Accuracy":accuracy_score(y_test, y_pred),
		"Precision": precision_score(y_test, y_pred),
		"Recall":recall_score(y_test, y_pred),
		"F1-Score":f1_score(y_test, y_pred),
		"ROC-AUC": roc_auc_score(y_test, y_fraud_prob),
		"PR-AUC": auc(recall, precision),
	}
    
    print("\n"+name.upper())
    print("-" * 30)
    for test_evaluation, value in results.items():
        if test_evaluation != 'Model':
            print(f"{test_evaluation}: \t{value:.5f}")
    print("\nConfusion Matrix:") 
    print(confusion_matrix(y_test, y_pred))
    print("-"*30)
    return results

