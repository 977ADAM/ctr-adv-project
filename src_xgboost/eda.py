# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(train_path):

    print("Loading data...")
    df = pd.read_parquet(train_path)

    print("\n====================")
    print("1. Target Distribution")
    print("====================")
    print(df["clicked"].value_counts())
    print("\nRatio:")
    print(df["clicked"].value_counts(normalize=True))

    sns.countplot(x="clicked", data=df)
    plt.title("Target Distribution")
    plt.show()

    print("\n====================")
    print("2. CTR by Hour")
    print("====================")
    hour_ctr = df.groupby("hour")["clicked"].mean()
    print(hour_ctr)

    hour_ctr.plot(kind="bar")
    plt.title("CTR by Hour")
    plt.ylabel("CTR")
    plt.show()

    print("\n====================")
    print("3. CTR by Day of Week")
    print("====================")
    dow_ctr = df.groupby("day_of_week")["clicked"].mean()
    print(dow_ctr)

    dow_ctr.plot(kind="bar")
    plt.title("CTR by Day of Week")
    plt.ylabel("CTR")
    plt.show()

    print("\n====================")
    print("4. Inventory CTR Variance")
    print("====================")
    inventory_ctr = df.groupby("inventory_id")["clicked"].mean()
    print(inventory_ctr.describe())

    print("\n====================")
    print("5. History Feature Correlation")
    print("====================")
    history_cols = [c for c in df.columns if c.startswith("history")]
    corr = df[history_cols + ["clicked"]].corr()["clicked"].sort_values(ascending=False)
    print(corr.head(10))


if __name__ == "__main__":
    run_eda("data/train.parquet")
