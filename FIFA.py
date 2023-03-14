from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('CompleteDataset.csv')


#exploracion de datos
print(data.head())

print(data.shape)

print(data.describe())



#division de datps
training, testing = train_test_split(data, test_size=0.2, random_state=42)


tuning, testing = train_test_split(testing, test_size=0.5, random_state=42)


