import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import sys
class KNNClassifier:
    def __init__(self, filepath, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        try:
            data = pd.read_csv(filepath).drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
            data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1}).astype(int)
            self.X = data.drop('diagnosis', axis=1).values
            self.y = data['diagnosis'].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42)
        except Exception as e:
            er_type, er_msg, line_no = sys.exc_info()
            print(f'Error in line no: {line_no.tb_lineno} and Error Message : {er_msg}')
    def evaluate(self, X, y_true):
        try:
         y_pred = self.model.predict(X)
         cm = np.zeros((2, 2))
         for true, pred in zip(y_true, y_pred):
             cm[true][pred] += 1
         tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
         accuracy = (tp + tn) / len(y_true)
         print(f"TP: {tp:.0f}, TN: {tn:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, Accuracy: {accuracy:.2f}")
        except Exception as e:
          er_type, er_msg, line_no = sys.exc_info()
          print(f'Error in line no: {line_no.tb_lineno} and Error Message : {er_msg}')
        return accuracy
    def optimize_k(self, k_range=range(3, 50, 2)):
        try:
         accuracies = [KNeighborsClassifier(n_neighbors=k).fit(self.X_train, self.y_train)
                       .score(self.X_test, self.y_test) for k in k_range]
         best_k = k_range[np.argmax(accuracies)]
         plt.figure(figsize=(8, 3))
         plt.plot(k_range, accuracies, 'r*-')
         plt.xlabel('k values')
         plt.ylabel('accuracy')
         plt.title('K-Value vs Accuracy')
         plt.show()
         print(f"Best k-value: {best_k} with accuracy: {max(accuracies):.2f}")
        except Exception as e:
          er_type, er_msg, line_no = sys.exc_info()
          print(f'Error in line no: {line_no.tb_lineno} and Error Message : {er_msg}')
        return best_k, accuracies
    def train_and_evaluate(self, k=None):
        try:
         if k:
             self.model = KNeighborsClassifier(n_neighbors=k)
         self.model.fit(self.X_train, self.y_train)
         print("\nTraining Performance:")
         self.evaluate(self.X_train, self.y_train)
         print("Test Performance:")
         self.evaluate(self.X_test, self.y_test)
        except Exception as e:
         er_type, er_msg, line_no = sys.exc_info()
         print(f'Error in line no: {line_no.tb_lineno} and Error Message : {er_msg}')
if __name__ == "__main__":
      try:
       knn = KNNClassifier('breast-cancer.csv')
       knn.train_and_evaluate()
       best_k, _ = knn.optimize_k()
       print(f"\nRefining model with best k-value: {best_k}")
       knn.train_and_evaluate(best_k)
      except Exception as e:
       er_type, er_msg, line_no = sys.exc_info()
       print(f'Error in line no: {line_no.tb_lineno} and Error Message : {er_msg}')