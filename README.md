# Credit Card Fraud Detection
Machine Learning Team Project - Topic: Fraud Detection

1. PROGRAMMING LANGUAGE AND VERSION
	-Language: Python
	-Version:  Python 3.10.12

2. Setup
	-Clone the repo and go into the folder:
		git clone https://github.com/SamikshaRijal/fraud-detection.git
		cd fraud-detection
	
	-Install the libraries:
		pip install -r requirements.txt

3. Dataset
	The dataset is not included in the repo.
	-Download it from Kaggle with link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
	- Unzip it and put 'creditcard.csv' in the 'data/' folder.

4. How to run
	- Change the current working directory to the project folder
	- python3 src/main.py	(run the model comparison and dataset summary)			
	- python3 src/preprocessing.py	(only runs the dataset summary)

5. Model tuning
    - Logistic Regression tests C = [0.01, 0.1, 1, 10].
    - Decision Tree tests max_depth = [5, 10, None].
    - Both models are tested with and without balanced class weights.
    - Settings are selected by mean PR-AUC (trapezoidal area) using 5-fold stratified cross-validation on the training split.
    - Time and Amount scaling is fitted inside each validation fold.
    - Selected models are refitted on the full training split, then evaluated on the held-out test split.
    - All trial scores are saved to results/metrics/tuning_results.csv.
    - Final test metrics and selected settings are saved to results/metrics/model_comparison.csv.
