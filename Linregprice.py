import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objects as go

cars = pd.read_csv('CarPrice_Assignment.csv')


df = cars.copy()
df.head()

#understanding columns
des = df.describe()
print(des)

#missing values
df.isnull().sum()

#split carname column
new = df["CarName"].str.split(" ", n=1, expand=True)
df['CarCompany']=new[0]
df['Carmodel']=new[1]
df.insert(2,'Company',new[0])
df.drop(['Carmodel','CarName','CarCompany'], axis=1, inplace=True)

df.Company.unique()

df.Company = df.Company.str.lower()

def replace(a,b):
    df.Company.replace(a,b,inplace=True)

replace('maxda', 'mazda')
replace('porcshce','porsche')
replace('toyouta','toyota')
replace('vokswagen','volkswagen')
replace('vw','volkswagen')

#plot prices
x=[df['price']]

#plt.figure(figsize=(10,8))
#plt.subplot(1,2,1)
#plt.hist(x, bins=50, color='red', rwidth=0.85)
#plt.subplot(1,2,2)
#plt.boxplot(x)
#plt.grid()



fig = go.Figure()
fig.add_trace(go.Histogram(x=df.price, name='control', marker_color='#EB89B5', opacity=0.75))

fig.update_layout(
    title_text='Sampled Results', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.3, # gap between bars of adjacent location coordinates
     # gap between bars of the same location coordinates
)
plot(fig)

#categorical data




