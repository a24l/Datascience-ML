import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('HR_comma_sep.csv')

df = data.copy()

df.head()


df.isnull().sum()

correlation = df.corr() 

left = df.left==1

retention = df.left==0


pd.crosstab(df.salary,df.left).plot(kind='bar')