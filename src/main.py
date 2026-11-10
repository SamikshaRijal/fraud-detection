import os
from pathlib import Path

import pandas as pd
from joblib import parallel_config
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from member3 import BalancedGradientBoostingClassifier, save_feature_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

from preprocessing import get_data
from eval import model_evaluation

# Use the same PR-AUC calculation for validation and final evaluation
def pr_auc(model, X, y):
    y_fraud_prob = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_fraud_prob)
    return auc(recall, precision)

def main():
    X_train, X_test, y_train, y_test = get_data(True, scaling=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        # Logistic regression
        "Logistic Regression (Balanced)":  LogisticRegression(class_weight = "balanced",max_iter=1000, random_state=42),
        "Logistic Regression (UnBalanced)":  LogisticRegression(max_iter=1000, random_state=42),
        # Decision Tree
        "Decision Tree (Balanced)": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Decision Tree (UnBalanced)": DecisionTreeClassifier(random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        "Random Forest (UnBalanced)": RandomForestClassifier(random_state=42, n_jobs=1),
        "Gradient Boosting (Balanced)": BalancedGradientBoostingClassifier(random_state=42),
        "Gradient Boosting (UnBalanced)": GradientBoostingClassifier(random_state=42),
    }

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    workers = max(1, min(4, int(os.environ.get("FRAUD_N_JOBS", "4"))))
    best_forest = None
    best_forest_score = float("-inf")
    results=[]
    tuning_results=[]
    confusion_results=[]
    for name, model in models.items():
        if isinstance(model, LogisticRegression):
            parameters = {"model__C": [0.01, 0.1, 1, 10]}
        elif isinstance(model, DecisionTreeClassifier):
            parameters = {"model__max_depth": [5, 10, None]}
        elif isinstance(model, RandomForestClassifier):
            parameters = {"model__n_estimators": [50, 100], "model__max_depth": [10, None]}
        elif isinstance(model, GradientBoostingClassifier):
            # Two joint candidates bound runtime; this is not an exhaustive search.
            parameters = [
                {"model__n_estimators": [50], "model__learning_rate": [0.1], "model__max_depth": [2]},
                {"model__n_estimators": [100], "model__learning_rate": [0.05], "model__max_depth": [3]},
            ]
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
            search = GridSearchCV(model, parameters, scoring=pr_auc, cv=cv, n_jobs=workers, pre_dispatch=workers, verbose=2, error_score="raise")
            with parallel_config(backend="loky", inner_max_num_threads=1):
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

        if isinstance(model.named_steps["model"], RandomForestClassifier):
            if search.best_score_ > best_forest_score:
                best_forest_score = search.best_score_
                best_forest = (model, name)

        result = model_evaluation(name, model, X_test, y_test)
        if parameters:
            result["Best Setting"] = str(search.best_params_)
            result["CV PR-AUC"] = search.best_score_
        results.append(result)
        tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1]).ravel()
        confusion_results.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        # Save completed models as checkpoints during the long ensemble run.
        pd.DataFrame(results).to_csv("results/metrics/model_comparison.csv", index=False)
        pd.DataFrame(confusion_results).to_csv("results/metrics/confusion_matrices.csv", index=False)
        if tuning_results:
            pd.concat(tuning_results, ignore_index=True).to_csv("results/metrics/tuning_results.csv", index=False)
        if best_forest is not None and name.startswith("Random Forest"):
            save_feature_importance(*best_forest)


    df_results = pd.DataFrame(results)
    print("\nModel Comparison: ")
    print("-"*30)
    print(df_results.to_string(index=False))
    df_results.to_csv("results/metrics/model_comparison.csv", index=False)
    if tuning_results:
        pd.concat(tuning_results, ignore_index=True).to_csv("results/metrics/tuning_results.csv", index=False)

    if best_forest is not None:
        importance = save_feature_importance(*best_forest)
        print("\nRandom Forest feature importance (selected by CV):")
        print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
