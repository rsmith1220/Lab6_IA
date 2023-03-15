from sklearn.model_selection import train_test_split
import pandas as pd
from data_explorer import DataExplorer

def proccess_data_lol():
    data = pd.read_csv('high_diamond_ranked_10min.csv')
    col = data.columns
    dae = DataExplorer(data, 'blueWins', value_split=True, keep_columns=['blueWins', "blueKills", "blueDeaths", "blueTowersDestroyed", "blueTotalGold", "redTowersDestroyed", "redTotalGold"])

if __name__ == "__main__":
    proccess_data_lol()    