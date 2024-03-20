import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import load_csv
import time

class LogisticRegression:
    def __init__(self, df, chart=False):
        self.df = self.stand_input_data(df)
        self.chart = chart
        if chart is True:
            self.show_chart()
    def stand_input_data(self, df):
        data = df.copy()
        self.means = {}
        self.std = {}
        self.means["theta0"] = 0
        self.std["theta0"] = 0
        for column_name, content in data.drop("Hogwarts House", axis=1, inplace=False).items():
            self.means[column_name] = 1 / content.size * content.values.sum()
            self.std[column_name] = (content.apply(lambda x: (x - self.means[column_name])**2).values.sum() / content.size)**0.5
            data[column_name] = data[column_name].apply(lambda x: (x - self.means[column_name]) / self.std[column_name])
        return data

    def training(self, step_size=0.09, training_iterations=1):
        ravenclaw_df = self.df.copy()
        ravenclaw_df.loc[ravenclaw_df['Hogwarts House'] == 'Ravenclaw', 'Hogwarts House'] = 1
        ravenclaw_df.loc[ravenclaw_df['Hogwarts House'] != 1, 'Hogwarts House'] = 0
        slytherin_df = self.df.copy()
        slytherin_df.loc[slytherin_df['Hogwarts House'] == 'Slytherin', 'Hogwarts House'] = 1
        slytherin_df.loc[slytherin_df['Hogwarts House'] != 1, 'Hogwarts House'] = 0
        gryffindor_df = self.df.copy()
        gryffindor_df.loc[gryffindor_df['Hogwarts House'] == 'Gryffindor', 'Hogwarts House'] = 1
        gryffindor_df.loc[gryffindor_df['Hogwarts House'] != 1, 'Hogwarts House'] = 0
        hufflepuff_df = self.df.copy()
        hufflepuff_df.loc[hufflepuff_df['Hogwarts House'] == 'Hufflepuff', 'Hogwarts House'] = 1
        hufflepuff_df.loc[hufflepuff_df['Hogwarts House'] != 1, 'Hogwarts House'] = 0
        ravenclaw_theta = self.binary_classification(ravenclaw_df, training_iterations, step_size)
        slytherin_theta = self.binary_classification(slytherin_df, training_iterations, step_size)
        gryffindor_theta = self.binary_classification(gryffindor_df, training_iterations, step_size)
        hufflepuff_theta = self.binary_classification(hufflepuff_df, training_iterations, step_size)
        self.store_parameters(ravenclaw_theta, slytherin_theta, gryffindor_theta, hufflepuff_theta)
    
    def binary_classification(self, df, training_iterations, step_size):
        Y = df["Hogwarts House"]
        X = df.drop("Hogwarts House", axis=1, inplace=False)
        theta_values = np.zeros(len(df.columns))
        temp_theta =  np.zeros(len(X.columns))
        temp_theta_bias = np.zeros(1)
        m = len(X.values)
        for _ in range(training_iterations):
            predictions = X.apply(lambda x: self.model_prediction(theta_values[1:].T, x.values, theta_values[0]), axis=1)
            temp_theta_bias = (1 / m) * (predictions - Y).sum()
            for i, (theta, column_name) in enumerate(zip(temp_theta, X)):
                calcul_sum = 0
                for j, (predict, value) in enumerate(zip(predictions, Y)):
                    calcul_sum += (predict - value) * X[column_name][j]
                derivative = (1 / m) * calcul_sum
                temp_theta[i] = derivative
            theta_values[1:] -= step_size * temp_theta
            theta_values[0] -= step_size * temp_theta_bias
            print(predictions)
        return theta_values

    def cost_function(self, fun_h):
        print(fun_h)

    def model_prediction(self, theta, x, bias):
        return self.sigmoid(bias + (theta * x).sum())

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def store_parameters(self, ravenclaw_theta, slytherin_theta, gryffindor_theta, hufflepuff_theta):
        data = {"Ravenclaw": ravenclaw_theta, "Slytherin": slytherin_theta, \
                "Gryffindor": gryffindor_theta, "Hufflepuff": hufflepuff_theta, \
                "mean": self.means.values(), "std": self.std.values()}
        theta_df = pd.DataFrame(data)
        theta_df.to_csv("parameters.csv", index=False)

    def show_chart(self):
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # x_data = [1, 2, 3]
        y_data = [4, 5, 6]

        # # Créer le graphique initial avec les données initiales
        # plt.plot(x_data, y_data, 'bo')
        # plt.show(block=False)
        #show
    
def parse_data(df):
    data = df.drop(["Index","First Name","Last Name","Birthday","Best Hand", "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1).replace([np.nan], 0)
    return data

def main():
    try:
        data_train = load_csv(sys.argv[1])
        data_parsed = parse_data(data_train)
        # x_data = [1, 2, 3]
        # y_data = [4, 5, 6]

        # Créer le graphique initial avec les données initiales
        # plt.plot(x_data, y_data, 'bo')  # Points bleus
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Mon graphique')
        # plt.grid(True)
        # plt.show(block=False)
        # input("Appuyez sur Entrée pour quitter...")  # Attendre l'entrée de l'utilisateur pour quitter

        logistic_regression = LogisticRegression(data_parsed, True)
        logistic_regression.training()

    except Exception as msg:
        print(msg, "Error")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith("dataset_train.csv") is True:
        main()
    else:
        print("Wrong argument: logreg_train.py dataset_train.csv")
