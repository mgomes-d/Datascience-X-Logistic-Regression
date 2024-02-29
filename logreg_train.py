import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np

class GradientDescent:
    def __init_(self):
        self.x = 0

class LogisticRegression:
    def __init__(self, df):
        self.x = 0
        self.df = stand_input_data(df)

    def stand_input_data(self, df):
        data = df
        return data


def load_csv(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path in wrong format, .csv"
    df = pd.read_csv(path)
    return df

def parse_data(df):
    data = df.drop(["Index","First Name","Last Name","Birthday","Best Hand", "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1).replace([np.nan], 0)
    return data

def main():
    try:
        # Charger les données
        data_train = load_csv(sys.argv[1])
        data_parsed = parse_data(data_train)
        logistic_regression = LogisticRegression()
        print(data_parsed)
        # logisticReg = LogisticRegression(data_parsed)

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith("dataset_train.csv") is True:
        main()
    else:
        print("Wrong argument: logreg_train.py dataset_train.csv")
