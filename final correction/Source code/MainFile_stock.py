#====================== IMPORT PACKAGES ==================================

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics
from sklearn import linear_model
from scipy.stats import pearsonr


#====================== READ INPUT DATA  ==================================

print("--------------------------------------")
print(" Input data ")
print("--------------------------------------")
print()
dataframe=pd.read_csv('Stock Dataset.csv')
print(dataframe.head(20))


#======================== PREPROCESSING  ==================================

#==== checking missing values =====

print("----------------------------------------------")
print(" Data Preprocessing ")
print("----------------------------------------------")
print()
print (dataframe.isnull().sum())


#==== cdrop unwanted columns =====

columns = ['Date']
dataframe.drop(columns, inplace=True, axis=1)

#========================== FEATURE SELECTION  =============================

# === Correlation ===


print("-------------------------------------------------------")
print("Correlation")
print("-------------------------------------------------------")
print()

val1 = dataframe['Close']
val2 = dataframe['High']
corr, _ = pearsonr(val1, val2)

print('Pearsons correlation :',  corr)
print()

#========================== DATA SPLITTING   =============================

#===== test and train ======

xx=dataframe.drop('Close',axis=1)
yy=dataframe['Close']

X_train, X_test, Y_train, Y_test = train_test_split(xx,yy,test_size=0.3,random_state=40)


print("----------------------------------------------")
print(" Data Splitting ")
print("----------------------------------------------")
print()
print("Total number of data's in input         :",dataframe.shape[0])
print() 
print("Total number of data's in training part :",X_train.shape[0])
print()
print("Total number of data's in testing part :",X_test.shape[0])
print()


#========================== CLASSIFICATION  =============================

# === SVR ===

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

svr_rbf.fit(X_train, Y_train)

y_rbf = svr_rbf.predict(X_test)

print("----------------------------------------------")
print(" Support Vector Regression ")
print("----------------------------------------------")
print()

Score_1=metrics.mean_absolute_error(Y_test, y_rbf)
print('1. Mean Absolute Error:', Score_1)  
print()
print('2. Mean Squared Error:', metrics.mean_squared_error(Y_test, y_rbf))  
print()
print('3. Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_rbf))) 
print()


# === LASSO REGRESSION ===

reg1 = linear_model.Lasso()
  
reg1.fit(X_train, Y_train)

prd_lrr=reg1.predict(X_test)

print("----------------------------------------------")
print(" Lasso Regression ")
print("----------------------------------------------")
print()

Score_2=metrics.mean_absolute_error(Y_test, prd_lrr)

print('1. Mean Absolute Error:', Score_2)  
print()
print('2. Mean Squared Error:', metrics.mean_squared_error(Y_test, prd_lrr))  
print()
print('3. Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, prd_lrr))) 
print()

#============================= PREDICTION  ================================

# === PREDICT THE STOCK PRICE ===

print("------------------------------------------------")
print("Prediction")
print("------------------------------------------------")
print()


for i in range(0,10):
    print("============================")
    print()
    print([i],"The stock price =",prd_lrr[i])


#============================= VISUALIZATION  ================================

# === prediction of stock ===

plt.title("Prediction of Stock")
plt.plot(prd_lrr) 
plt.show() 


print()
print("------------------------------------------------------------")
print()


# === comparison ===

import matplotlib.pyplot as plt
import numpy as np


objects = ('SVR', 'Lasso Regression')
y_pos = np.arange(len(objects))
performance = [Score_1,Score_2]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance ')
plt.title('Comparison Graph -- Error Values')
plt.show()













