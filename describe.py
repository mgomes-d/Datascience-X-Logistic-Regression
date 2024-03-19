import sys
import csv
import pandas as pd
import numpy as np
from utils import load_csv

class Describe:
    def __init__(self, path: str):
        self.df: pd.DataFrame = load_csv(path)
        self.features = self.get_features()

    def get_features(self):
        return self.df.select_dtypes(include=[float])

    def get_all_information(self):
        options = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        features_names = []
        values = []
        for feature in self.features:
            features_names.append(feature)
            values.append(self.get_feature_informations(self.features[feature].replace([np.nan], 0).values.astype(float)))
        data = {}
        for feature_name, value in zip(features_names, values):
            data[feature_name] = value
        all_information = pd.DataFrame(data, index=options)
        print(all_information)


    def get_feature_informations(self, values_calcul):
        values_return = []
        values_return.append(self.count(values_calcul))
        values_return.append(self.mean(values_calcul))
        values_return.append(self.std(values_calcul))
        values_return.append(self.min(values_calcul))
        values_return.append(self.percentile(values_calcul, 0.25))
        values_return.append(self.percentile(values_calcul, 0.50))
        values_return.append(self.percentile(values_calcul, 0.75))
        values_return.append(self.max(values_calcul))
        return values_return
        
    def max(self, values):
        maximun = values[0]
        for value in values:
            if value > maximun:
                maximun = value
        return maximun
    
    def mean(self, values):
        add_values = 0
        for value in values:
            if value:
                add_values += value
        return add_values / len(values)

    def std(self, values):
        mean = self.mean(values)
        add_values = 0
        for value in values:
            add_values += (value - mean)**2
        std = (add_values / len(values))**0.5
        return std

    def min(self, values):
        minimun = values[0]
        for value in values:
            if minimun > value:
                minimun = value
        return minimun

    def percentile(self, values, percent):
        sorted_values = np.sort(values)
        percentile = round(len(values) * percent)
        return sorted_values[percentile]

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
