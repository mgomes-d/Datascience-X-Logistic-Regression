import pandas as pd


def load_file(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path need to end with .csv"
    df = pd.read_csv(path)
    return df

def prediction(parameters_df, train_data_df):
    print(parameters_df)
    print(train_data_df)

def main():
    try:
        parameters_df = load_file("parameters.csv")
        train_data_df = load_file("datasets/dataset_test.csv")
        prediction(parameters_df, train_data_df)
        # predict = df["theta0"].values.astype(float) + (df["theta1"].values.astype(float) * float(value))
        # print("Estimate price =",predict[0])
    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    main()
