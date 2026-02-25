# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm-

Import Libraries: Bring in the necessary libraries.

2.Load the Dataset: Load the dataset into your environment.

3.Data Preprocessing: Handle any missing data and encode categorical variables as needed.

4.Define Features and Target: Split the dataset into features (X) and the target variable (y).

5.Split Data: Divide the dataset into training and testing sets.

6.Build Multiple Linear Regression Model: Initialize and create a multiple linear regression model.

7.Train the Model: Fit the model to the training data.

8.Evaluate Performance: Assess the model's performance using cross-validation.

9.Display Model Parameters: Output the model’s coefficients and intercept.

10.Make Predictions & Compare: Predict outcomes and compare them to the actual values.
## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: A PRAVEEN KISHORE
RegisterNumber:  212225220074


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

data= pd.read_csv('CarPrice_Assignment.csv')
print(data.head())


data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)


X=data.drop('price',axis=1)
y=data['price']
X_train , X_test, y_train , y_test = train_test_split(X,y,test_size=0.2 ,random_state=42)


model= LinearRegression()
model.fit(X_train , y_train)

print('Name:A PRAVEEN KISHORE')
print('Reg No: 25010010')
print('\n=== Cross-validation ===')
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R2 :",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")


y_pred = model.predict(X_test)
print("\n=== Test set Performance ===")
print(f"MSE : {mean_squared_error(y_test,y_pred):.2f}")
print(f"R2 : {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")


plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
*/
```

## Output:
<img width="768" height="176" alt="1" src="https://github.com/user-attachments/assets/6ed0b3fd-1983-480c-bc5a-4d986b672ad0" />
<img width="407" height="123" alt="2" src="https://github.com/user-attachments/assets/1e806f95-69da-4c12-b624-d59da4059e7e" />
<img width="1102" height="690" alt="3" src="https://github.com/user-attachments/assets/8493952a-91d6-4270-a81f-206ad61c60c8" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
