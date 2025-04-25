# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### step 1: Import the required packages and print the present data.
### step 2: Print the placement data and salary data.
### step 3: Find the null and duplicate values.
### step 4: Using logistic regression find the predicted values of accuracy , confusion matrices.
### step 5: Display the results.

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: KARTHICK S

RegisterNumber: 212224230114
*/

```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()
```

## output:
![ex 5 s1](https://github.com/user-attachments/assets/2289e107-2d63-4504-ae5f-cfee83e1ca75)

```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
## output:
![ex 5 s2](https://github.com/user-attachments/assets/9d085cbd-7c7a-4e43-9903-b8cfaa3bd84d)

```
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1
```
## output:
![ex 5 s3](https://github.com/user-attachments/assets/11f4c027-5d1e-46bb-833a-dd8799288e69)

```
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
```

## output:
![ex 5 s4](https://github.com/user-attachments/assets/c2ae6f64-c8ea-41a6-b83f-4b478fc9f0e4)


```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
## output:
![ex 5 s5](https://github.com/user-attachments/assets/7a0b4165-4acd-4290-ba1f-d5337268daa4)


```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
## output:
![ex 5 s6](https://github.com/user-attachments/assets/91e213d4-7191-42bd-85f1-213f7232b08b)

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## output:

![ex 5 s7](https://github.com/user-attachments/assets/8cfeec2d-a590-409b-8c1c-5574b91bc66f)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
