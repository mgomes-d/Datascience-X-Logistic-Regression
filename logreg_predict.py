import pandas as pd
import numpy as np


class Predict_price:
    def __init__(self, parameters_df, predict_data_df):
        self.parameters_df = parameters_df
        self.predict_data_df = predict_data_df
        self.predict_data_df.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand', "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1, inplace=True)
        self.predict_data_df.fillna(0, inplace=True)
        self.normalize_data()

    def normalize_data(self):
        self.means = self.parameters_df["mean"].values.astype(float)[1:].copy()
        self.std = self.parameters_df["std"].values.astype(float)[1:].copy()
        del(self.parameters_df["mean"])
        del(self.parameters_df["std"])
        for i, (column_name, values) in enumerate(self.predict_data_df.items()):
            self.predict_data_df[column_name] = values.apply(lambda x: (x - self.means[i]) / self.std[i])

    # def denormalize_data(self)


    def prediction(self):
        for column, theta_value in self.parameters_df.items():
            # print(theta_value)
            self.binary_classification(theta_value.values.astype(float), self.predict_data_df.iloc[0].values.astype(float))
            break

        # print(predict_data_df.iloc[0].values.astype(float))

    # def predict_one_value()

    def binary_classification(self, theta, values):
        test= do after at campus
        result = sigmoid(test)

def load_file(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path need to end with .csv"
    df = pd.read_csv(path)
    return df

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    try:
        parameters_df = load_file("parameters.csv")
        predict_data_df = load_file("datasets/dataset_test.csv")
        prediction = Predict_price(parameters_df, predict_data_df)
        prediction.prediction()
        # predict = df["theta0"].values.astype(float) + (df["theta1"].values.astype(float) * float(value))
        # print("Estimate price =",predict[0])
    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    main()
