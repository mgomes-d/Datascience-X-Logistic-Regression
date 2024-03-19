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
            sig_prediction = sigmoid(sum_prediction)
            result.append(sig_prediction)
        return result

    def create_file(self, list_predict):
        df = pd.DataFrame(list_predict, columns=['Hogwarts House'])
        df.rename_axis('Index', inplace=True)
        df.to_csv("houses.csv", index=True)

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
        predictions_values = prediction.prediction()
        prediction.create_file(predictions_values)


    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    main()
