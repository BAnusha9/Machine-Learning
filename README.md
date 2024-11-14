# Machine Learning
# KNN 
# Breast Cancer Identification using K-Nearest Neighbors Algorithm (KNN)

 Breast cancer develops in breast cells, and as per statistics of 2019 in the U.S., About 1 in 8 U.S. women (about 12%) will develop invasive breast cancer throughout their lifetime. Breast cancer is an equally critical disease in dogs and cats: It is the most common tumor found in female dogs and the third most prevalent tumor detected in cats. So the early diagnosis is crucial for survival. It is important to note that most breast lumps are benign and not cancer (malignant). Non-cancerous breast tumors are abnormal growths but do not spread outside the breast. So another big challenge is to identify if the cancer lumps are malignant or benign. This article shows how the machine learning approach KNN can be used for this identification task.
 
# K-Nearest Neighbors Algorithm

K-Nearest Neighbors Algorithm is one of the most simple and easily interpretable supervised machine learning algorithm. One specialty of K-NN is that, it does not have a separate training phase. The algorithm takes the whole data as training set. . KNN can be used to solve both classification and regression problems, however, it is generally used to solve classification problems.


![KNN](https://github.com/user-attachments/assets/266e504b-876f-4962-8e9a-e020627d3096)

 Assume we have to determine the class of “?”. The K in KNN means how many near by neighbors we wish to consider for voting for “?”. When K=3, we consider three adjacent datapoints we have 2 out of 3 data points as blue circle so When K=3, new point will be classified as blue circle. In the same way, In case of K=7 out of 7 points 4 adjacent datapoints are of class green circle. So the new data point will be classified to green circle.

To summarize this, KNN simply calculates the distance of an unknown/new data point to all other training data points. The Metric generally used for the distance calculation are Euclidean, Manhattan etc. It then selects the K-nearest data points and assigns the new data point the class to which the majority of the K data points belong.

Dataset

In this classification task, we use the Breast cancer wisconsin (diagnostic) dataset to predict whether the cancer is benign or malignant. The dataset features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The data has only two labels Malignant(M) or Benign(B).

Importing libraries and Reading Data - Code

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv("breast_cancer.csv")

df.head()

The top 5 rows of the dataframe are displayed as below.

![1659874322440](https://github.com/user-attachments/assets/c38497b2-53c0-4eb8-b3b4-d70c07c09762)


The diagnosis column is our target variable and you can notice that we have one unwanted column with all NaN values ‘Unnamed:32’. Also note that the ID column has no significance, So, we can remove both ID and Unnamed:32 columns.

df=df.drop(['id','Unnamed: 32'],axis=1)

check whether any of the columns contain null values

df.isnull().sum()
The data does not contain missing values. Which means the data is very clean and polished. The label values are ‘M’ and ‘B’ corresponding to the Malignant and Benign classes. We can convert them to 0 and 1 respectively.

df['diagnosis'] = df['diagnosis'].map({'M':0,'B':1}).astype(int)

Creating the KNN Model

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier 

Before feeding the data to the algorithm, we split the data into labels and features.

X = df.iloc[: , 1:]
y = df.iloc[: , 0]

For evaluating the model we have to take train and test datasets separately.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
 we need to give the training data to algorithm [knn]
so knn will break the data into 2 parts
 M related patients part and B related patients part

 selecting Best K_value
k_values = np.arange(3,50,2)
test_accuracy_ = []

for i in k_values:
  reg = KNeighborsClassifier(n_neighbors=i)
  reg.fit(X_train,y_train)
  test_accuracy_.append(reg.score(X_test,y_test))

  test_accuracy_
  the output is given as 
  [0.9298245614035088,
 0.956140350877193,
 0.956140350877193,
 0.956140350877193,
 0.9824561403508771,
 0.9736842105263158,
 0.9649122807017544,
 0.9649122807017544,
 0.9649122807017544,
 0.9649122807017544,
 0.956140350877193,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315,
 0.9473684210526315]

 test_accuracy_.index(max(test_accuracy_))
 4
 
 finding the best k value 
 k_values[test_accuracy_.index(max(test_accuracy_))] 
 output : 11
 hence,the best k valuse is 11
We can visualize the scores for each value of k, by the below plot using matplotlib

plt.figure(figsize=(8,3))

plt.xlabel('k_values')

plt.ylabel('accuracy_values')

plt.plot(k_values,test_accuracy_,color = 'r',marker='*')

plt.show()

![Figure_1](https://github.com/user-attachments/assets/5abde8f6-4c71-41bc-adaf-36fa055e8f84)

Creating the Model with best k value 

from sklearn.neighbors import KNeighborsClassifier

reg = KNeighborsClassifier(n_neighbors=11) # default k value is = 5

reg.fit(X_train,y_train)

output :  KNeighborsClassifier?i

KNeighborsClassifier(n_neighbors=11)

# Confusion Matrix, Accuracy, Precision, Recall, F1-Score

![download](https://github.com/user-attachments/assets/156938ef-6ee6-4963-b1b8-cee4ecc8f022)

Introduction

Imagine you’re training a spam filter. How do you measure the performance of the model? Is it more important to correctly identify all actual spam emails, even if it mistakenly flags some legitimate emails, or vice versa? In this blog post, we’ll learn how to use a tool called confusion matrix and its derived metrics to evaluate the performance of classification models.

# Confusion Matrix

A confusion matrix is a fundamental tool for evaluating the performance of classification models in machine learning. It’s a simple table that visualizes how often your model correctly or incorrectly predicts the various classes (or categories) within your dataset.

Key Components

True Positives (TP): The number of instances your model correctly predicted as positive.

True Negatives (TN): The number of instances your model correctly predicted as negative.

False Positives (FP): The number of instances your model incorrectly predicted as positive (also known as Type I error).

False Negatives (FN): The number of instances your model incorrectly predicted as negative (also known as Type II error).

Accuracy

Accuracy is one of the most basic metrics used to evaluate a classification model. It represents the percentage of correct predictions made by your model. To calculate accuracy from a confusion matrix, you use the following formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

In essence, accuracy tells you what proportion of total predictions (both positive and negative) were correctly classified by your model.

While accuracy is a valuable metric, it’s essential to recognize that it can be misleading if you have a dataset with imbalanced classes (significantly more of one class than the other). Let’s say you have 1000 emails in your inbox. Only 10 of these are spam, while 990 are legitimate emails. A simple spam filter that classifies everything as “not spam” would achieve 99% accuracy. However, this filter would be terrible because it fails to catch any of the actual spam emails, which is its primary purpose. In such cases, you’ll want to consider additional metrics like precision, recall, and F1-score for a more complete picture of your model’s performance.

Precision

The formula for precision based on a confusion matrix:

Precision = TP / (TP + FP)

Precision measures the proportion of true positive predictions among all positive predictions. In our spam filter example, precision tells us what percentage of emails flagged as spam were actually spam.

High precision: Great! Your model rarely flags innocent emails as spam.
Low precision: Oops! Your model is firing off false alarms, flagging many legitimate emails as spam.

Recall

The formula for recall based on a confusion matrix:

Recall = TP / (TP + FN)

Recall measures the proportion of true positive predictions among all actual positive instances. For the spam filter, recall tells us what percentage of actual spam emails were correctly identified.

High recall: Fantastic! Your model catches most of the spam emails.
Low recall: Uh oh! Your model is letting some spam slip through the cracks.

F1-Score

Precision and recall often have an inverse relationship. Optimizing for one might come at the expense of the other. Imagine tightening your spam filter’s criteria to improve precision (fewer false alarms). This might also decrease recall (missing more actual spam).

So, which one matters more? It depends! In healthcare, where misdiagnoses are critical, high recall might be paramount. In finance, where false positives can trigger unnecessary transactions, high precision might be crucial. Consider the real-world implications of your model’s predictions to weigh the importance of each metric.

Sometimes, finding the right balance between precision and recall is crucial. That’s where the F1-score comes in. F1-score is a harmonic mean of precision and recall, calculated as:

F1 = 2 * (precision * recall) / (precision + recall)
F1-score:

Ranges from 0 to 1, with 1 being the best score.
Combines the strengths of precision and recall into a single metric.
Useful when a balanced evaluation of both aspects is needed.




Model Performance

actual | Predict

1 | 1 = TP

0 | 0 = TN

0 | 1 = FP

1 | 0 = FN

Accuracy

`tp + tn` / `tp + tn + fp + fn `


train performance 

tp = 0

tn = 0

fp = 0

fn = 0


for i in training_data.index:

  if training_data['y_train_values'][i] == 1 and training_data['y_train_pred_values'][i] == 1:

    tp = tp + 1
  elif training_data['y_train_values'][i] == 0 and training_data['y_train_pred_values'][i] == 0:

    tn = tn + 1
  elif training_data['y_train_values'][i] == 0 and training_data['y_train_pred_values'][i] == 1:

    fp = fp + 1

  else:

    fn = fn + 1


print(f'True Positive Count : {tp}')

print(f'True Negative Count : {tn}')

print(f'False Positive Count : {fp}')

print(f'False Negative Count : {fn}')


output: True Positive Count : 277

True Negative Count : 147

False Positive Count : 22

False Negative Count : 9


Train Accuracy

(tp + tn) / (tp+tn+fp+fn)

0.9318681318681319


Evaluating the Model

y_test_pred = reg.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,

classification_report
confusion_matrix(y_train,y_train_pred)
array([[147,  22],
       [  9, 277]])

accuracy_score(y_train,y_train_pred)
0.9318681318681319

print(classification_report(y_train,y_train_pred))



               precision    recall  f1-score   support

            0       0.94      0.87      0.90       169
            1       0.93      0.97      0.95       286
     accuracy                           0.93       455
    macro avg       0.93      0.92      0.93       455
 weighted avg       0.93      0.93      0.93       455

Test_Performence

y_test_pred = reg.predict(X_test) 
confusion_matrix(y_test,y_test_pred)
output : array([[41,  2],
               [ 0, 71]])

accuracy_score(y_test,y_test_pred)

0.9824561403508771

print(classification_report(y_test,y_test_pred))


              precision    recall  f1-score   support

           0       1.00      0.95      0.98        43
           1       0.97      1.00      0.99        71
      accuracy                           0.98       114
      macro avg       0.99      0.98      0.98       114
     weighted avg       0.98      0.98      0.98       114


The model performance is decent for classification of breast cancer.You can change the parameters and data preprocessing and always improve the results. As mentioned earlier, it is important to note that K-Nearest Neighbors Algorithm doesn’t always perform as well with high-dimensionality or categorical features. The main objective of this task is to introduce the simple yet powerful ML algorithm KNN with a use case.




