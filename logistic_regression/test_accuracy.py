from sklearn.metrics import accuracy_score
from utils.utils import load_csv
from logistic_regression.logreg_predict import Predict_price

def main():
    dataset_train = load_csv("datasets/dataset_train.csv")
    dataframe_parameters = load_csv("parameters.csv")
    dataset_result = dataset_train['Hogwarts House']
    dict_house = {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
    # dataset_train.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand', "Potions", "Arithmancy", "Care of Magical Creatures"], axis=1, inplace=True)
    # dataset_train.fillna(0, inplace=True)
    predictions = Predict_price(dataframe_parameters, dataset_train)
    predictions_values = predictions.prediction()
    predictions_index = [dict_house[house] for house in predictions_values]
    real_values_index = [dict_house[house] for house in dataset_result]
    accuracy = accuracy_score(real_values_index, predictions_index)
    print(accuracy)
    # print(predictions_index)

if __name__ == "__main__":
    main()
