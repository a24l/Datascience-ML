import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


data1 = pd.read_csv('bottle.csv')

df = data1[['Salnty', 'T_degC']]

df.isnull().sum()

df.dropna(axis=0, inplace=True)

Features = list(['Salnty'])
Target = list(['T_degC']) 

x = df.loc[:, Features].astype(float)
y= df.loc[:, Target].astype(float)


# split data into train and test 
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=0)

LinR = LinearRegression(train_x, train_y)

model = LinR.fit(train_x,train_y)

model.coef_
model.intercept_