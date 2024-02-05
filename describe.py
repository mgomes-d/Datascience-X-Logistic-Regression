import sys
import csv
import pandas as pd

def load_csv(path: str):
    df = pd.read_csv(path)
    return df

def main():
    try:
        df = load_csv(sys.argv[1])
        print(df.dtypes)

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: describe.py dataset.csv")
