import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Add import for models that you are implementing
from preprocessing import get_data
from eval import model_evaluation

X_train, X_test, y_train, y_test = get_data(True)
    
models = {
    # Logistic regression
	"Logistic Regression (Balanced)":  LogisticRegression(max_iter=1000, random_state=42),
	"Logistic Regression (UnBalanced)":  LogisticRegression(class_weight = "balanced", max_iter=1000, random_state=42),
	# Decision Tree
	"Decision Tree (Balanced)": DecisionTreeClassifier(random_state=42),
	"Decision Tree (UnBalanced)": DecisionTreeClassifier(class_weight="balanced", random_state=42),
}

results=[]
for name, model in models.items():
    model.fit(X_train, y_train)
    results.append(model_evaluation(name, model, X_test, y_test))


df_results = pd.DataFrame(results)
print("\nModel Comparison: ")
print("-"*30)
print(df_results.sort_values(by="Accuracy", ascending=False))