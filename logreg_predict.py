import pandas as pd
import numpy as np


def load_file(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path need to end with .csv"
    df = pd.read_csv(path)
    return df

def prediction(parameters_df, predict_data_df):
    # print(parameters_df)
    predict_data_df.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand'], axis=1, inplace=True)
    predict_data_df.fillna(0, inplace=True)
    print(parameters_df["Gryffindor"])
    # print(predict_data_df)

# def predict_one_value()

def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def main():
    try:
        parameters_df = load_file("parameters.csv")
        predict_data_df = load_file("datasets/dataset_test.csv")
        prediction(parameters_df, predict_data_df)
        # predict = df["theta0"].values.astype(float) + (df["theta1"].values.astype(float) * float(value))
        # print("Estimate price =",predict[0])
    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    main()
