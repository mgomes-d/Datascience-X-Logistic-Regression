import pandas as pd
import numpy as np
import sys
from utils.utils import load_csv
import threading

class LogisticRegression:
    def __init__(self, df):
        self.df = self.stand_input_data(df)
        self.ravenclaw_df = self.parse_house("Ravenclaw")
        self.slytherin_df = self.parse_house("Slytherin")
        self.gryffindor_df = self.parse_house("Gryffindor")
        self.hufflepuff_df = self.parse_house("Hufflepuff")
        self.ravenclaw_theta = np.zeros(len(self.df.columns))
        self.slytherin_theta = np.zeros(len(self.df.columns))
        self.gryffindor_theta = np.zeros(len(self.df.columns))
        self.hufflepuff_theta = np.zeros(len(self.df.columns))
        self.ravenclaw_loss = []
        self.slytherin_loss = []
        self.gryffindor_loss = []
        self.hufflepuff_loss = []

    def parse_house(self, house_name):
        clean_df = self.df.copy()
        clean_df.loc[clean_df['Hogwarts House'] == house_name, 'Hogwarts House'] = 1
        clean_df.loc[clean_df['Hogwarts House'] != 1, 'Hogwarts House'] = 0
        return clean_df

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

    def train_house(self, house_name, step_size, training_iterations):
        theta_values = getattr(self, house_name.lower() + "_theta")
        df = getattr(self, house_name.lower() + "_df")
        theta_values = self.binary_classification(house_name, df, training_iterations, step_size, theta_values)
        setattr(self, house_name.lower() + "_theta", theta_values)

    def training(self, house_name: str, step_size=0.00000000001, training_iterations=10):
        all_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
        for house in all_houses:
            if house_name == house:
                self.train_house(house, step_size, training_iterations)
    
    def binary_classification(self, house_name, df, training_iterations, step_size, theta_values):
        Y = df["Hogwarts House"]
        X = df.drop("Hogwarts House", axis=1, inplace=False)
        temp_theta =  np.zeros(len(X.columns))
        temp_theta_bias = np.zeros(1)
        m = len(X.values)
        loss_name = house_name.lower() + "_loss"
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
            getattr(self, loss_name).append(self.cost_function(predictions, Y, m))
        return theta_values

    def cost_function(self, predictions, real_values, m):
        cost = -(1/m) * (real_values * np.log(predictions) + (1 - real_values) * np.log(1 - predictions)).sum()
        return cost

    def get_cost(self, house_name):
        return getattr(self, house_name.lower() + "_loss")

    def get_cost_and_reset(self, house_name):
        values = getattr(self, house_name.lower() + "_loss")
        setattr(self, house_name.lower() + "_loss", [])
        return values

    def model_prediction(self, theta, x, bias):
        return self.sigmoid(bias + (theta * x).sum())

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def store_parameters(self):
        data = {"Ravenclaw": self.ravenclaw_theta, "Slytherin": self.slytherin_theta, \
                "Gryffindor": self.gryffindor_theta, "Hufflepuff": self.hufflepuff_theta, \
                "mean": self.means.values(), "std": self.std.values()}
        theta_df = pd.DataFrame(data)
        theta_df.to_csv("logistic_regression/parameters.csv", index=False)

def parse_data(df):
    data = df.drop(["Index","First Name","Last Name","Birthday","Best Hand", "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1)
    means = []
    for content in data.drop("Hogwarts House", axis=1, inplace=False):
        means.append(np.mean(data[content].dropna().values))
    
    for content, mean in zip(data.drop("Hogwarts House", axis=1, inplace=False), means):
        data[content].replace(np.nan, mean, inplace=True)
    return data

def main():
    try:
        data_train = load_csv(sys.argv[1])
        data_parsed = parse_data(data_train)
        logistic_regression = LogisticRegression(data_parsed)
        threads = []
        all_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
        for house in all_houses:
            thread = threading.Thread(target=logistic_regression.training, args=(house,))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        logistic_regression.store_parameters()

    except Exception as msg:
        print(msg, "Error")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith("dataset_train.csv") is True:
        main()
    else:
        print("Wrong argument: logreg_train.py dataset_train.csv")
