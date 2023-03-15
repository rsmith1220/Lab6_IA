import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import re

class DataExplorer():

    def __init__(self, data: DataFrame, target, random_state = 2, test_size = 0.3, value_split = False, drop_columns = [], keep_columns = []) -> None:
        self.data = data
        self.target = target
        self.random_state = random_state
        self.test_size = test_size
        self.value_split = value_split
        # Primero elige las columans de interes
        if not keep_columns:
            self.drop_columns(drop_columns, reinstantiateXY = False)
        else:
            self.data = self.data[keep_columns]
        # Elimina filas con valores nulos
        self.clean_data_none(reinstantiateXY = False)
        # Separa el dataframe en X y y
        self.calculate_X_and_Y() #Instantiate self.X and self.Y
        # Separa la data en datos de prueba
        self.calculate_split_data()#Instantiate self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val

    def reinstantiate_x_y(self):
        self.calculate_X_and_Y()
        return self.calculate_split_data()

    def calculate_X_and_Y(self):
        self.X = self.data.loc[:,self.data.columns!=self.target]
        self.Y = self.data.loc[:,self.data.columns==self.target]
        return self.X, self.Y

    def calculate_split_data(self):
        self.X_val, self.Y_val = None, None
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = self.test_size, random_state = self.random_state)
        if self.value_split:    
            self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(self.X_test, self.Y_test, test_size=0.5, random_state = self.random_state)
        return self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val 

    def clean_data_none(self, reinstantiateXY = False):
        initial_shape = self.data.shape
        self.data = self.data.dropna()
        final_shape = self.data.shape
        print("Original dataframe shape: ", initial_shape)
        print("Cleaned dataframe shape: ", final_shape)
        remaining_data = final_shape[0]*100/initial_shape[0]
        print(f"Remaining data: {remaining_data}")
        if reinstantiateXY: self.reinstantiate_x_y()

    def correlation_in_dataset(self, relation_min = 0.7):
        def __increase_or_add(dicc, element):
            if element in dicc:
                dicc[element] += 1
            else: 
                dicc[element] = 1
        
        if not 0<relation_min<=1: relation_min = 0.7
        corr_df = self.data.corr(method="pearson")# Matriz de correlaciona a analizar
        high_corr = set()
        correlations_count = {}
        for i in range(len(corr_df.columns)):
            for j in range(i):
                correlation = abs(corr_df.iloc[i, j])
                if correlation > relation_min: #Eligo las variables con correlacion 
                    colname = corr_df.columns[i]
                    colname_related = corr_df.columns[j]
                    high_corr.add((colname, colname_related, correlation))
                    __increase_or_add(correlations_count, colname)
                    __increase_or_add(correlations_count, colname_related)
        return high_corr, correlations_count        

    def drop_columns(self, columns, reinstantiateXY = True):
        if not columns: return
        if self.target in columns:
            print(f"Cannot delete target column: {self.target}")
            return
        self.data = self.data.drop(columns=columns)
        if reinstantiateXY: self.reinstantiate_x_y()

    def find_columns_with_regex(self, pattern):
        matches = self.data.applymap(lambda x: bool(re.search(pattern, str(x)))).any()
        return matches[matches == True].index.tolist()
    
    def apply_func_column(self, func, column, astype, reinstantiateXY = False):
        self.data[column] = self.data[column].apply(func).astype(astype)
        if reinstantiateXY: self.reinstantiate_x_y()
    
    def appliable_func_value_on_condition(self, column, operation, parameter):
        operations = {
            "==": lambda x: x==parameter,
            ">=": lambda x: x>=parameter, 
            ">" : lambda x: x>parameter,
            "<=": lambda x: x<=parameter, 
            "<" : lambda x: x<parameter
        }
        if operation not in operations:
            key_str = ', '.join(str(key) for key in operations.keys())
            print(f"Must be one of theese: {key_str}")
            return
        return self.apply_func_column(operations[operation], column, "float", False)

    def accuracy(self, y_predict):
        y_ref = list(self.Y_test[self.target])
        if len(y_predict)!=len(y_ref):
            print("Prediction must be the same legth as the test. Try predicting with X_test")
            return None
        acc = 0
        test_size = len(y_ref)
        for i in range(test_size):
            if y_ref[i]==y_predict[i]:
                acc += 1
        return acc/test_size
    