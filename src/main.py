import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc

# Add import for models that you are implementing
from preprocessing import get_data
from eval import model_evaluation

# Use the same PR-AUC calculation for validation and final evaluation
def pr_auc(model, X, y):
    y_fraud_prob = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_fraud_prob)
    return auc(recall, precision)

X_train, X_test, y_train, y_test = get_data(True, scaling=False)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
models = {
    # Logistic regression
	"Logistic Regression (Balanced)":  LogisticRegression(class_weight = "balanced",max_iter=1000, random_state=42),
	"Logistic Regression (UnBalanced)":  LogisticRegression(max_iter=1000, random_state=42),
	# Decision Tree
	"Decision Tree (Balanced)": DecisionTreeClassifier(class_weight="balanced", random_state=42),
	"Decision Tree (UnBalanced)": DecisionTreeClassifier(random_state=42),
}

results=[]
tuning_results=[]
for name, model in models.items():
    if isinstance(model, LogisticRegression):
        parameters = {"model__C": [0.01, 0.1, 1, 10]}
    elif isinstance(model, DecisionTreeClassifier):
        parameters = {"model__max_depth": [5, 10, None]}
    else:
        parameters = {}

    # Scale inside each fold and keep the original feature order
    scale = ColumnTransformer([
        ("time", StandardScaler(), ["Time"]),
        ("features", "passthrough", X_train.columns.drop(["Time", "Amount"])),
        ("amount", StandardScaler(), ["Amount"]),
    ])
    model = Pipeline([("scale", scale), ("model", model)])

    if parameters:
        print("\nTuning " + name)
        search = GridSearchCV(model, parameters, scoring=pr_auc, cv=cv, n_jobs=-1, verbose=1, error_score="raise")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best parameters:", search.best_params_)
        print(f"CV PR-AUC: {search.best_score_:.5f}")

        scores = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
        scores.insert(0, "Model", name)
        tuning_results.append(scores)
        print(scores.to_string(index=False))
    else:
        model.fit(X_train, y_train)

    result = model_evaluation(name, model, X_test, y_test)
    if parameters:
        result["Best Setting"] = str(search.best_params_)
        result["CV PR-AUC"] = search.best_score_
    results.append(result)


df_results = pd.DataFrame(results)
print("\nModel Comparison: ")
print("-"*30)
print(df_results.to_string(index=False))
df_results.to_csv("results/metrics/model_comparison.csv", index=False)
if tuning_results:
    pd.concat(tuning_results, ignore_index=True).to_csv("results/metrics/tuning_results.csv", index=False)
