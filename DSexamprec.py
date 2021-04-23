import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

df = pd.read_csv('apy.csv')
df.head()

#null values check
df.isnull().sum()
df.dropna(inplace=True)

#Q1 - Which of the following statements is/are TRUE? 
#The overall production during the Kharif season was 4,029,970,000 (approx.)
# The overall production during the Summer season was 434,549,800 (approx.) 
#The overall production during the Autumn season was 64,413,770 (approx.)
# The overall production during the Kharif season was 2,051,688,000 (approx.)

df.groupby(['Season'])['Production'].sum()

#Q2 - Which of the following crops were produced during the Kharif season?
df['Season'] = df['Season'].str.strip()
df[df['Season'] == 'Kharif']['Crop'].unique()


#Q3 - Which district in India had the highest crop production?
df['District_Name'] = df['District_Name'].str.strip()
df.sort_values(by='Production', ascending=False).head()

df1 = df.groupby(['District_Name'])['Production'].sum().reset_index()
df1.sort_values(by='Production', ascending=False)

#Q4 - During which year, Tamil Nadu had the highest crop production?
df1 = df[df['State_Name'] == 'Tamil Nadu']

#df.groupby(['State_Name'])['Production'].sum()
df1.groupby(['Crop_Year'])['Production'].sum().reset_index().sort_values(by='Production', ascending=False)


#Q5 - Which state in India had the highest crop production? (overall, for all years)?
df.groupby('State_Name')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head()


#Q6 - What is the relation between Area and Production?
correlation = df.corr()

#Q7 - What is the average crop production?
#Q8 - What is the standard deviation of production?
df['Production'].describe()

#Q9 - Due to some unknown reasons, the crop production for only two states were recorded for the year 2015. Which states are they?
df[df['Crop_Year'] == 2015]['State_Name'].unique()

#Q10 - The top three produced crops in the year 1997
df[df['Crop_Year'] == 1997].groupby('Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head()

#Q12 - Which year had the highest crop production? (overall)
df.groupby('Crop_Year')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head()

#Q13 - The crop that Tamil Nadu produced the most was
df[df['State_Name'] == 'Tamil Nadu'].groupby('Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)

# Data Model
df2 = df.drop(['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop'], axis=1, inplace=True) 

X = df[['Area']]

y = df[['Production']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 111)

lnR = LinearRegression(fit_intercept=True)

model = lnR.fit(X_train, y_train)

pred = lnR.predict(X_test)


#R-Squared model
rsq = lnR.score(X_test, y_test)

print(rsq)

lnR_mse = mean_squared_error(y_test, pred)

lnR_rmse = np.sqrt(lnR_mse)
print(lnR_rmse)
