from data_explorer import DataExplorer
import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None) -> None:
        #For decision node -> Nodo de decision >> Nodo que contiene condición
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain 
        # For leaf node -> Hoja con valor >> El valor de la decision, la clase seleccionada para el nodo
        self.value = value

class DecisionTreeClassifierLP():
    def __init__(self, dae: DataExplorer, min_samples_split= 2, max_depth = 2) -> None:
        self.root =  None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.dae: DataExplorer = dae
        self.X_train, self.X_test, self.X_val, self.Y_train, self.Y_test, self.Y_val = self.dae.calculate_split_data()
    
    def build_tree(self, dataset, curr_depth = 0):
        """ Metodo recursivo para construir árbol """
        x, y = dataset[:, :-1], dataset[:, -1] 
        num_samples, num_featues = np.shape(x)
        #split until stopping condition are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.__get_best_split(num_samples, num_featues, dataset)
            #Check if info gain is positive
            if best_split["info_gain"]>0:
                #recurs left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)

                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)

                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])
        leaf_value = self.__calculate_leaf_value(y)
        return Node(value=leaf_value)
    def fit(self):
        x = self.X_train
        y = self.Y_train
        dataset = pd.concat([x, y], axis= 1).values #usamos el dataset completo como numpy array
        self.root = self.build_tree(dataset)

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            if isinstance(x, pd.DataFrame):
                x = x.values
            else:
                print("Not acceptable type of x in prediction")
                return None

        return [self.make_prediction(i, self.root) for i in x]
         
    def make_prediction(self, x, tree: Node):
        if tree.value != None:  return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        return self.make_prediction(x, tree.right)

    def __get_best_split(self, num_samples, num_features, dataset):
        best_split = {"info_gain": -float("inf")}
        max_info_gain = -float('inf')
        for feature_index in range(num_features):
            # get current split
            feature_values = dataset[:, feature_index]
            possible_threshold = self.__find_possibles_threshold(feature_values) # Todas las posbilidades alm
            for threshold in possible_threshold:
                dataset_left, dataset_right = self.__split(dataset, feature_index, threshold)
                if len(dataset_left)>9 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.__information_gain(y, left_y, right_y, "gini")  
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] =  feature_index
                        best_split["threshold"] =  threshold 
                        best_split["dataset_left"]=  dataset_left
                        best_split["dataset_right"]= dataset_right
                        best_split["info_gain"]= curr_info_gain       
                        max_info_gain = curr_info_gain        
        return best_split

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


    def __split(self, dataset, feature_index, threshold):
        dataset_left = np.empty((0, dataset.shape[1]))
        dataset_right = np.empty((0, dataset.shape[1]))
        for row in dataset:
            if row[feature_index] <= threshold:
                dataset_left = np.vstack((dataset_left, row))
            else:
                dataset_right = np.vstack((dataset_right, row))
        return dataset_left, dataset_right
    
    def __information_gain(self, parent, left_child, right_child, mode= "entropy"):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        if mode=="gini": 
            return self.__gini_index(parent) - (weight_left*self.__gini_index(left_child) + weight_right * self.__gini_index(right_child))
        else: #Entropy:
            return self.__entropy(parent) - (weight_left*self.__entropy(left_child) + weight_right * self.__entropy(right_child))

    def __entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy 

    def __gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            gini += p_cls**2
        return 1- gini

    def __calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)
    
#Pruebas
if __name__ == "__main__":
    data = pd.read_csv('high_diamond_ranked_10min.csv')
    col = data.columns
    dae = DataExplorer(data, 'blueWins', value_split=True, keep_columns=['blueWins', "blueKills", "blueDeaths", "blueTowersDestroyed", "blueTotalGold", "redTowersDestroyed", "redTotalGold"])
    dae.reinstantiate_x_y()
    dtc = DecisionTreeClassifierLP(dae, max_depth=5)
    dtc.fit()
    x_test = dae.X_test
    y_predict = dtc.predict(x_test)    
    acc = dae.accuracy(y_predict)
    print(acc)
        

        