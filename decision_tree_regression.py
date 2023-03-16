from data_explorer import DataExplorer
import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, var_red = None, value = None) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red

        self.value = value

class DecisionTreeRegressor():
    
    def __init__(self, dae: DataExplorer, min_samples_split = 2, max_depth = 2) -> None:
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.dae = dae
        self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val = self.dae.calculate_split_data()

    def build_tree(self, dataset, curr_depth = 0):
        x, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features  = np.shape(x)
        best_split = {}
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.__get_best_split(dataset, num_samples, num_features)
            if best_split["var_red"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["var_red"])
        leaf_value = self.__calculate_leaf_value(y)
        return Node(value= leaf_value)
    
    def __variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child)
        weight_r = len(r_child)
        return np.var(parent) - weight_l * np.var(l_child) - weight_r * np.var(r_child)
        
    def __calculate_leaf_value(self, y):
        return np.mean(y)
    
    def __split(self, dataset, feature_index, threshold):
        dataset_left = np.empty((0, dataset.shape[1]))
        dataset_right = np.empty((0, dataset.shape[1]))
        for row in dataset:
            if row[feature_index] <= threshold:
                dataset_left = np.vstack((dataset_left, row))
            else:
                dataset_right = np.vstack((dataset_right, row))
        return dataset_left, dataset_right
    
    def __find_possibles_threshold(self, featured_values, limit = 100):
        """Make sures i dont get to much value of features being 100 the max. 

        Args:
            featured_values (_type_): _description_
            limit (int, optional): _description_. Defaults to 100.
        """
        possible_threshold = np.unique(featured_values)
        if possible_threshold.size<=limit: 
            return possible_threshold
        return np.percentile(possible_threshold, range(1, 101))
    
    def __get_best_split(self, dataset, num_samples, num_features):
        best_split = {"var_red": -float("inf")}
        max_var_red = -float("inf")
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_threshold = self.__find_possibles_threshold(feature_values)
            for threshold in possible_threshold:
                dataset_left, dataset_right = self.__split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_var_red = self.__variance_reduction(y, left_y, right_y)
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] =  feature_index
                        best_split["threshold"] =  threshold 
                        best_split["dataset_left"]=  dataset_left
                        best_split["dataset_right"]= dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
        return best_split

    def fit(self):
        x = self.X_train
        y = self.Y_train
        dataset = pd.concat([x, y], axis= 1).values #usamos el dataset completo como numpy array
        self.root = self.build_tree(dataset)
    
    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold: 
            return self.make_prediction(x, tree.left)
        return self.make_prediction(x, tree.right)
    
    def predict(self, x):
        if not isinstance(x, np.ndarray) or isinstance(x, list):
            if isinstance(x, pd.DataFrame):
                x = x.values
            else:
                print("Not acceptable type of x in prediction")
                return None
        return [self.make_prediction(i, self.root) for i in x]
        