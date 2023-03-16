from sklearn.model_selection import train_test_split
import pandas as pd
from data_explorer import DataExplorer
from decision_tree_classifier import DecisionTreeClassifierLP


# Import the necessary libraries
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def DTC_scikit(X_train, X_test, X_val, Y_train, Y_test, Y_val):
    clf = DecisionTreeClassifier(max_depth=5)

    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    print("\tAccuracy of LOL BlueWins prediction of DTC from scikit: ", accuracy)

def proccess_data_lol():
    data = pd.read_csv('high_diamond_ranked_10min.csv')
    dae = DataExplorer(data, 'blueWins', value_split=True, keep_columns=['blueWins', "blueKills", "blueDeaths", "blueTowersDestroyed", "blueTotalGold", "redTowersDestroyed", "redTotalGold"])
    X_train, X_test, X_val, Y_train, Y_test, Y_val = dae.reinstantiate_x_y()
    dtc = DecisionTreeClassifierLP(dae, max_depth=5)
    dtc.fit()
    x_test = dae.X_test
    y_predict = dtc.predict(x_test)    
    acc = dae.accuracy(y_predict)
    print("LOL results: ")
    print(f"\tAccuracy of LOL BlueWins prediction of DTC from scratch: {acc}")
    DTC_scikit(X_train, X_test, X_val, Y_train, Y_test, Y_val)
    
if __name__ == "__main__":
    proccess_data_lol()    