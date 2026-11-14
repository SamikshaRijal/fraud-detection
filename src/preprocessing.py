import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(summary=False, scaling=True):
    file = "data/creditcard.csv"
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        raise ValueError(f"File '{file}' not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"'{file}' CSV file is empty.")
    
    if summary: 
        print("\tDATASET SUMMARY")
        print("-" * 30)
        print("Number of observations(rows): ", df.shape[0])
        print(f"Features: {df.shape[1]-1} + 1 Class Label")
        print("Missing Values:", df.isnull().sum().sum())
        print("Duplicate data rows: ", df.duplicated().sum())
        counts = df["Class"].value_counts()
        print("\nClass Counts: ")
        print(f"Normal (0): {counts[0]}")
        print(f" Fraud (1): {counts[1]}")
        print(f"Percent of normal charges: {counts[0]/len(df) * 100:.3f}%")
        print(f" Percent of fraud charges: {counts[1]/len(df) * 100:.3f}%")

    
    # Remove all the duplicate data rows
    df = df.drop_duplicates()
    
    # Separate features and target class label
    X = df.drop(columns=["Class"])
    y= df["Class"]
    
    # Split the dataset into 80/20 training and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
    X_train = X_train.copy()
    X_test = X_test.copy()

    #Scale time and amount and fit on training set
    if scaling:
        scale = StandardScaler()
        X_train[['Time', 'Amount']]= scale.fit_transform(X_train[['Time', 'Amount']])
        X_test[['Time', 'Amount']] = scale.transform(X_test[['Time', 'Amount']])
    
    if summary:
        print("\nAfter the split: ")
        print("-" * 30)
        print(f"Training set: Total= {len(X_train)}, Fraud= {y_train.sum()}")
        print(f" Testing set: Total= {len(X_test)}, Fraud= {y_test.sum()}")
      
    return X_train, X_test, y_train, y_test
    

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(True)
    
    