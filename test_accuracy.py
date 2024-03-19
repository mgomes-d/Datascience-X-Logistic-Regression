from sklearn.metrics import accuracy_score
from utils import load_csv
from logreg_predict import Predict_price


def load_csv(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path in wrong format, .csv"
    df = pd.read_csv(path)
    return df

def main():
    print("test")
    #main

if __name__ == "__main__":
    main()
