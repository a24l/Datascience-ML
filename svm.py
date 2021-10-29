import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics, svm
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


X1 = pd.read_csv('irisX.csv')
Y1 = pd.read_csv('irisY.csv')
X= X1.copy()
Y= Y1.copy()

Y.head()
X.head()


#rename/ add the row in x datafile
newrow= pd.DataFrame({'5.2': 5.2, '2.7':2.7, '3.9':3.9, '1.4':1.4}, index=[0])                   
# simply concatenate both dataframes
X_new = pd.concat([newrow, X ]).reset_index(drop = True)
X_new.rename(columns = {'5.2': 'A', '2.7':'B', '3.9':'C', '1.4':'D'}, inplace=True)
X_Drop= X_new.drop(['C', 'D'], axis=1)

#rename/ add the row in y datafile
newrowy= pd.DataFrame({'1': 1}, index=[0])
Y_new = pd.concat([newrowy, Y ]).reset_index(drop = True)
Y_new.rename(columns = {'1': 'A'}, inplace=True)

#slicing train data
X_train = X_Drop[0:100]
Y_train = Y_new[0:100]

#slicing test data
X_test = X_Drop[100:]
Y_test = Y_new[100:]


#### Logistic Regression
clf = LogisticRegression(penalty = 'l2', C=1e4, multi_class = 'ovr')
clf.fit(X_train, Y_train)
pred= clf.predict(X_test)
#find accuracy for LR
acc= clf.score(X_test,Y_test)



#### SVM classification
svmclf= svm.SVC(kernel='rbf', gamma=0.5, C=1.0, decision_function_shape = 'ovr')
svmclf.fit(X_train, Y_train)
# find accuracy for SVM
svmacc= svmclf.score(X_test,Y_test)
#total n0. of support vectors
suppvec= svmclf.n_support_

#classification report
print(classification_report(Y_test,pred))

#confusion matrix
cm= confusion_matrix(Y_test, pred)
