# Credit Card Fraud Detection
CSE-4309: Machine Learning Team Project - Topic: Fraud Detection
Team: Samiksha Rijal, Sagar Khadka, Bipin Adhikari, Samin Rijal, Bardan Karki

This folder contains all the code needed to reproduce the results in our report.

1. PROGRAMMING LANGUAGE AND VERSION
	-Language: Python
	-Version:  Python 3.10.12
	-Libraries: pandas, numpy, scikit-learn, matplotlib (listed in `requirements.txt`)

	Install the libraries from inside the project folder in bash:
		>pip install -r requirements.txt


2. Dataset
	The dataset is not included due to its size.
	-Download it from Kaggle with link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
	- Unzip it and put 'creditcard.csv' in the 'data/' folder.

4. How to run
	- Change the current working directory to the project folder
	Run all commands from the project folder.

	1. Train and evaluate all models
		>python3 src/main.py	
	This prints the dataset summary and the results of all 9 model configurations, then saves the comparison tables, the feature importance chart and the trained models. It takes a long time to finish, mostly because of Gradient Boosting and KNN.
	
	2. Run the demo
		>python3 src/demo.py
	Choose a model (1–9). The demo loads the saved model and predicts 2 normal and 2 fraud transactions from the test set, showing the actual class, the predicted class and the fraud probability.

	Also:
	- python3 src/explore_dataset.py   # to understand the dataset and get class distribution chart
	- python3 src/preprocessing.py     # dataset summary and train/test split sizes only

4. Files

	-src/preprocessing.py: Loads the data, removes duplicates, makes a stratified 80/20 split, scales Time and Amount
|	-src/eval.py: Chooses each model's decision threshold from the training data and calculates the evaluation metrics
	-src/balancedGradient.py: Balanced version of Gradient Boosting (sample weights, since it has no class_weight option)
	-src/main.py: Trains and evaluates all models and saves the results 
	-src/demo.py: Live demo using a saved trained model 
	-src/explore_dataset.py: Learn the dataset and class distribution chart

## 5. Outputs
	- results/comparison_default_threshold.csv: Test results of all models at the default 0.5 threshold 
	-results/comparison_tuned_threshold.csv: Test results of all models at their tuned thresholds
	-results/final_output.txt- Full printed output of the run used in the report
	-results/figures/feature_importance.png: Top 15 features of the balanced Random Forest
	-results/figures/class_distribution.png: Normal vs. fraud transaction counts 
	-results/models/*.pkl: Saved trained models with their thresholds (created by main.py)

All results use random_state = 42.