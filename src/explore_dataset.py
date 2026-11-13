# Creates a class distribution chart for the report and the slides
# Run: python3 src/explore_dataset.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


df = pd.read_csv("data/creditcard.csv")

print("Number of observations(rows): ", df.shape[0])
print("Features: ", df.shape[1]-1)

print("Missing Values:", df.isnull().sum().sum())
print("Duplicate data rows: ", df.duplicated().sum())

counts = df["Class"].value_counts()
print("Class Counts: ", counts)
print(f"Percent of fraud charges: {counts[1]/len(df) * 100}%")
print(f"Percent of normal charges: {counts[0]/len(df) * 100}%")

os.makedirs("results/figures", exist_ok=True)
# Bar chart showing the number of normal vs fraud transaction
# Since there are fewer fraud transactions log scale is used
counts.plot(kind="bar")
plt.yscale("log")
plt.xlabel('Class (Normal = 0, Fraud = 1)')
plt.ylabel('Number of transactions')
plt.title('Fraud vs Normal Transactions')
plt.savefig("results/figures/class_distribution.png")
plt.close()

