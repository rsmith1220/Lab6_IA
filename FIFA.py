from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from data_explorer import DataExplorer

def convert_money_to_number(x):
    factor = 1
    if 'K' in x:
        factor = 1e3
    elif 'M' in x:
        factor = 1e6
    if factor!=1: return float(x[1:-1]) * factor
    else: return float(x[1:])

def convert_to_number_or_operation(x):
    """Apply to column: Acceleration, Agility, Ball control, Composure

    Args:
        x (str): Value in acceleration column

    Returns:
        Numeric:  Value of either the operation contained or the number value. 
    """
    if '+' in x:
        return sum([int(num) for num in x.split('+')])
    elif '-' in x:
        nums = [int(num) for num in x.split('-')]
        return nums[0] - sum(nums[1:])
    else:
        return pd.to_numeric(x)

data = pd.read_csv('CompleteDataset.csv')
dae = DataExplorer(data, "Potential", keep_columns=['Age', 'Overall', 'Value', 'Wage', 'Special', 'Acceleration', 'Agility', 'Ball control', 'Composure', 'Potential'])
data = dae.data
# Getting number value from each column 
dae.apply_func_column(convert_money_to_number, "Value", "float") #Value -> Money
dae.apply_func_column(convert_money_to_number, "Wage", "float")#Wage -> money
dae.apply_func_column(convert_to_number_or_operation, "Acceleration", "float")#Acceleration -> Nummber or operation
dae.apply_func_column(convert_to_number_or_operation, "Agility", "float")#Agility -> Number or operation
dae.apply_func_column(convert_to_number_or_operation, "Ball control", "float")#Ball control -> Number or operation
dae.apply_func_column(convert_to_number_or_operation, "Composure", "float")#Composure
dae.clean_data_none()
dae.reinstantiate_x_y()

high_corr, correlations_count = dae.correlation_in_dataset(0.7) #
print('a')

