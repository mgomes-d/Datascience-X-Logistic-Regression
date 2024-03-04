import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np

# class GradientDescent:
#     def __init_(self):
#         self.x = 0

class LogisticRegression:
    def __init__(self, df):
        self.df = self.stand_input_data(df)

    def stand_input_data(self, df):
        data = df.copy()
        self.means = {}
        self.std = {}
        for column_name, content in data.drop("Hogwarts House", axis=1, inplace=False).items():
            self.means[column_name] = 1 / content.size * content.values.sum()
            self.std[column_name] = (content.apply(lambda x: (x - self.means[column_name])**2).values.sum() / content.size)**0.5
            data[column_name] = data[column_name].apply(lambda x: (x - self.means[column_name]) / self.std[column_name])
        return data

    def training(self, training_step=0.01, training_iterations=10):
        for _ in range(training_iterations):
            print("hell")
            # print()
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

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
        # print(data_parsed)
        logistic_regression = LogisticRegression(data_parsed)
        logistic_regression.training()

    except Exception as msg:
        print(msg, "ddf")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith("dataset_train.csv") is True:
        main()
    else:
        print("Wrong argument: logreg_train.py dataset_train.csv")
