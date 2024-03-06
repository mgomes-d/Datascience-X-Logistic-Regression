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
        # data.loc[data['Hogwarts House'] == 'Ravenclaw', 'Hogwarts House'] = 0
        # data.loc[data['Hogwarts House'] == 'Slytherin', 'Hogwarts House'] = 1
        # data.loc[data['Hogwarts House'] == 'Gryffindor', 'Hogwarts House'] = 2
        # data.loc[data['Hogwarts House'] == 'Hufflepuff', 'Hogwarts House'] = 3
        # print(data)
        return data

    def training(self, step_size=0.01, training_iterations=1):
        theta_values = np.zeros(len(self.df.columns) - 1)
        for _ in range(training_iterations):
            # print(self.df)
            ravenclaw_df = self.df.copy()
            ravenclaw_df.loc[ravenclaw_df['Hogwarts House'] == 'Ravenclaw', 'Hogwarts House'] = 1
            ravenclaw_df.loc[ravenclaw_df['Hogwarts House'] != 1, 'Hogwarts House'] = 0
            self.binary_classification(ravenclaw_df, theta_values)
            # print(ravenclaw_df)
            # loss_value = (1 / len(df.columns)) * (ravenclaw_df.apply()).sum()
        print("loss_value")
    
    def binary_classification(self, df, theta_values):
        Y = df["Hogwarts House"]
        X = df.drop("Hogwarts House", axis=1, inplace=False)
        temp_theta = theta_values.copy()
        m = len(X.values)
        print(m)
        predictions = X.apply(lambda x: self.model_prediction(theta_values.T, x.values), axis=1)
        for i, (theta, column_name) in enumerate(zip(temp_theta, X)):
            calcul_sum = 0
            for j, (predict, value) in enumerate(zip(predictions, Y)):
                calcul_sum += (predict - value) * X[column_name][j]
            print((1 / m) * calcul_sum)
            derivative = (1 / m) * calcul_sum
            temp_theta[i] = derivative

        print(temp_theta)

    def model_prediction(self, theta, x):
        return self.sigmoid((theta * x).sum())

    def sigmoid(self, x):
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
        print(msg, "Error")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith("dataset_train.csv") is True:
        main()
    else:
        print("Wrong argument: logreg_train.py dataset_train.csv")
