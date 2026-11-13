import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from balancedGradient import BalancedGradientBoostingClassifier
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add import for models that you are implementing
from preprocessing import get_data
from eval import model_evaluation, find_best_threshold

X_train, X_test, y_train, y_test = get_data(True)

models = {
    # Logistic regression
    "Logistic Regression (Balanced)":  LogisticRegression(class_weight = "balanced", max_iter=1000, random_state=42),
	"Logistic Regression (UnBalanced)":  LogisticRegression(max_iter=1000, random_state=42),
 
	# Decision Tree
    "Decision Tree (Balanced)": DecisionTreeClassifier(class_weight="balanced", random_state=42),
	"Decision Tree (UnBalanced)": DecisionTreeClassifier(random_state=42),

 
    # Random Forest 
    "Random Forest (Balanced)": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    "Random Forest (UnBalanced)": RandomForestClassifier(random_state=42, n_jobs=-1),
    
    # Gradient Boosting
    "Gradient Boosting (Balanced)": BalancedGradientBoostingClassifier(random_state=42),
    "Gradient Boosting (UnBalanced)": GradientBoostingClassifier(random_state=42),
}

os.makedirs("results/models", exist_ok=True)

default_threshold_results=[]
tuned_threshold_results=[]
for name, model in models.items():
    print(f"\nTraining {name}...")
    threshold = find_best_threshold(model, X_train, y_train)
    model.fit(X_train, y_train)
    
    default_threshold_results.append(model_evaluation(name, model, X_test, y_test))
    
    tuned_threshold_results.append(model_evaluation(name, model, X_test, y_test, threshold))
    fileName = os.path.join("results/models", name.lower() + ".pkl")
    with open(fileName, "wb") as f:
        pickle.dump({"name": name, "model": model, "threshold":threshold}, f)
        
    if name == "Random Forest (Balanced)":
        importance = model.feature_importances_
        featureNames = X_train.columns 
        importance_df = pd.DataFrame({
            'Feature': featureNames,
            'Importance': importance
        }).sort_values(by='Importance', ascending = False)
        
        top15=importance_df.head(15)
        plt.figure(figsize=(8, 5))
        plt.barh(top15['Feature'], top15['Importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel('Importance Score')
        plt.title('Random Forest Feature Importance for top 15 features')
        plt.tight_layout()
        os.makedirs("results/figures", exist_ok=True)
        plt.savefig("results/figures/feature_importance.png", dpi=150)
        plt.close()

df_results = pd.DataFrame(default_threshold_results)
print("\nModel Comparison (Default Threshold(0.5)): ")
print("-"*30)
print(df_results.sort_values(by="F1-Score", ascending=False))

df_results = pd.DataFrame(tuned_threshold_results)
print(f"\nModel Comparison (Tuned Threshold): ")
print("-"*30)
print(df_results.sort_values(by="F1-Score", ascending=False))