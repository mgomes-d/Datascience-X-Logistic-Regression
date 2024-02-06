import sys
import csv
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    assert path.lower().endswith(".csv"), "Path in wrong format, .csv"
    df = pd.read_csv(path)
    return df


class Describe:
    def __init__(self, path: str):
        self.df: pd.DataFrame = load_csv(path)
        self.features = self.get_features()

    def get_features(self):
        return self.df.select_dtypes(include=[float])

    def get_all_information(self):
        options = ["Count", "Mean", "Std", "25%", "50%", "75%", "Max"]
        for feature in self.features:
            values = self.get_feature_informations(self.features[feature].values.astype(float))
            # print(feature)
        # data = {'':options}
        # print(self)
        # all_information = pd.DataFrame(data)
        # print(all_information)


    def get_feature_informations(self, values_calcul):
        values_return = []
        values_return.append(self.count(values_calcul))
        values_return.append(self.mean(values_calcul))
        print(values_return)
        
    
    def mean(self, values):
        add_values = 0
        print(values)
        for value in values:
            add_values += value
        
        return add_values / len(values)

    def count(self, values):
        return len(values)
    
    

def main():
    try:
        describe = Describe(sys.argv[1])
        describe.get_all_information()

    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
    else:
        print("Wrong argument: describe.py dataset.csv")
