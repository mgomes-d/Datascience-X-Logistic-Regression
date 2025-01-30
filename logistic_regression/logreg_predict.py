import pandas as pd
import numpy as np
from utils.utils import load_csv

class Predict_house:
    def __init__(self, parameters_df, predict_data_df):
        self.parameters_df = parameters_df
        self.predict_data_df = predict_data_df
        self.predict_data_df.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand', "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1, inplace=True)
        self.remove_nan()
        self.normalize_data()

    def remove_nan(self):
        means = []
        for content in self.predict_data_df:
            means.append(np.mean(self.predict_data_df[content].dropna().values))

        for content, mean in zip(self.predict_data_df, means):
            self.predict_data_df[content].replace(np.nan, mean, inplace=True)

    def normalize_data(self):
        self.means = self.parameters_df["mean"].values.astype(float)[1:].copy()
        self.std = self.parameters_df["std"].values.astype(float)[1:].copy()
        del(self.parameters_df["mean"])
        del(self.parameters_df["std"])
        for i, (column_name, values) in enumerate(self.predict_data_df.items()):
            self.predict_data_df[column_name] = values.apply(lambda x: (x - self.means[i]) / self.std[i])

    def prediction(self):
        predictions = {}
        for column, theta_value in self.parameters_df.items():
            predictions[column] = self.binary_classification(theta_value.values.astype(float), self.predict_data_df.values.astype(float))

        list_of_lists = [list(values) for values in predictions.values()]
        list_of_lists = list(map(list, zip(*list_of_lists)))
        predict_index = []
        for values in list_of_lists:
            predict_index.append(np.argmax(values))
        list_of_houses = list(predictions.keys())
        predict_str = []
        for value in predict_index:
            predict_str.append(list_of_houses[value])
        return predict_str

    def binary_classification(self, theta, values):
        result = []
        for value in values:
            sum_prediction = theta[0] + (theta[1:] * value).sum()
            sig_prediction = self.sigmoid(sum_prediction)
            result.append(sig_prediction)
        return result

    def create_file(self, list_predict):
        df = pd.DataFrame(list_predict, columns=['Hogwarts House'])
        df.rename_axis('Index', inplace=True)
        df.to_csv("logistic_regression/houses.csv", index=True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def main():
    try:
        parameters_df = load_csv("logistic_regression/parameters.csv")
        predict_data_df = load_csv("datasets/dataset_test.csv")
        prediction = Predict_house(parameters_df, predict_data_df)
        predictions_values = prediction.prediction()
        prediction.create_file(predictions_values)


    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    main()
