# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:03:57 2023

@author: lenovo
"""


1] Problem

BUSINESS OBJECTIVE:
    
1) Delivery_time -> Predict delivery time using sorting time..(It is Target Variables)
    
    

#Importing the Necessary Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from scipy import stats
import pylab

#Loading the dataset

df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/360DIGITMG ASSIGNMENT/simple linear regression/datasets/delivery_time.csv')
df.info()
df.describe()
#data organizing(Rename the columns)
df=df.rename(columns={'Delivery Time':'delivt','Sorting Time':'sortt'})

#To check whether data is normal or not we use,shapiro test
stats.shapiro(df.delivt)
pvalue=0.8963273763656616>0.05:-Data is normal
stats.shapiro(df.sortt)
pvalue=0.1881045252084732>0.05:-Data is normal

#To check Normality,using graphical visulisation(Univarite Analysis)
stats.probplot(df.delivt,plot=pylab)
stats.probplot(df.sortt,plot=pylab)

#For Bivariate Analysis,we use Scatter plot

plt.scatter(df.sortt,df.delivt,color='green')

#To quantify above,we use Co-relation of Coefficent(r)

np.corrcoef(df.sortt,df.delivt)
r=0.82599726 which tell us it is moderate(close to strong) linear relationship.

#For Varience,we Calculate CO-Varience
np.cov(df.sortt,df.delivt)

#Import Liabrary
import statsmodels.formula.api as smf

#Simple Linear Regression

model=smf.ols('delivt ~ sortt',data=df).fit()
model.summary()

#Predictions for this best fit line

pred=model.predict(pd.DataFrame(df['sortt']))

#Finding RMSE(Error Calculation)

error=df.delivt - pred #ACTUAL VALUE - PREDICTED VALUE
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line

plt.scatter(df.sortt,df.delivt)
plt.plot(df.sortt,pred,'r')
plt.legend(['predicted line','observed data'])
plt.show()

#we do transformations because we got,
Rsquare(Coefficent of determination=0.68)
so we need to capture more varience...
other factors including (r) and (RMSE) values are fine..


#Firstly we do Log Transformation

model1=smf.ols('delivt ~ np.log(sortt)',data=df).fit()
model1.summary()

#Predictions for this best fit line

pred1=model1.predict(pd.DataFrame(df['sortt']))

#Finding RMSE(Error Calculation)

error=df.delivt - pred1 #ACTUAL VALUE - PREDICTED VALUE
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line

plt.scatter(df.sortt,df.delivt)
plt.plot(df.sortt,pred1,'r')
plt.legend(['predicted line','observed data'])
plt.show()

# Secondly we do Exponantial Transformation

model2=smf.ols('np.log(delivt) ~ sortt',data=df).fit()
model2.summary()

#Predictions for this best fit line

pred2=model2.predict(pd.DataFrame(df['sortt']))
pred2_exp=np.exp(pred2)
pred2_exp
#Finding RMSE(Error Calculation)

error=df.delivt - pred2_exp #ACTUAL VALUE - PREDICTED VALUE
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line

plt.scatter(df.sortt,df.delivt)
plt.plot(df.sortt,pred2_exp,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#Then square Transformations

model3=smf.ols('delivt ~ (sortt*sortt)',data=df).fit()
model3.summary()

#Predictions for this best fit line

pred3=model3.predict(pd.DataFrame(df['sortt']))

#Finding RMSE(Error Calculation)

error=df.delivt - pred3 #ACTUAL VALUE - PREDICTED VALUE
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line

plt.scatter(df.sortt,df.delivt)
plt.plot(df.sortt,pred3,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#fourth we do Square Root Transforamtions

model4=smf.ols('delivt ~ np.sqrt(sortt)',data=df).fit()
model4.summary()

#Predictions for this best fit line

pred4=model4.predict(pd.DataFrame(df['sortt']))

#Finding RMSE(Error Calculation)

error=df.delivt - pred4 #ACTUAL VALUE - PREDICTED VALUE
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line

plt.scatter(df.sortt,df.delivt)
plt.plot(df.sortt,pred4,'r')
plt.legend(['predicted line','observed data'])
plt.show





#Lastly we do Polynominal Transformation


model5=smf.ols('np.log(delivt) ~ sortt + I(sortt*sortt)',data=df).fit()
model5.summary()
#we got  R-squared:=0.765.which is good..
Hence we use this model...
but also check all other factors is it fine or not?

pred5=model5.predict(pd.DataFrame(df['sortt']))

#Finding RMSE(Error Calculation)

error=df.delivt - pred5 #ACTUAL VALUE - PREDICTED VALUE
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#The Best Model

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2)

finalmodel=smf.ols('np.log(delivt) ~ sortt + I(sortt*sortt)',data=train).fit()
finalmodel.summary()

#Predictions for test data
testpred=finalmodel.predict(pd.DataFrame(test))
testpred_exp=np.exp(testpred)

#RMSE for test data

error=df.delivt - testpred_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Predictions for Train data
trainpred=finalmodel.predict(pd.DataFrame(train))
trainpred_exp=np.exp(trainpred)

#RMSE for Train data

error=df.delivt - trainpred_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse




2]Problem
    
BUSINESS OBJECTIVE:-

Salary_hike -> Build a prediction model for Salary_hike
::--Here the input variable is years of exprience and output/Target Variable is Salary   

#LODING THE DATASET
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/360DIGITMG ASSIGNMENT/simple linear regression/datasets/Salary_Data.csv')
df.info()
df.describe()

#Rename the Columns
df=df.rename(columns={'YearsExperience':'yexp','Salary':'sal'})
#TO Check normality,we use Shapiro Test
stats.shapiro(df.yexp)
stats.shapiro(df.sal)

#check noramlity using Graphical Representation(Univariate Anaylsis)
stats.probplot(df.yexp,plot=pylab)
stats.probplot(df.sal,plot=pylab)

#Checking Normality using Bivariate Analysis,we use Scatter Plot(It is Subjective)
plt.scatter(df.yexp,df.sal)

#To quantify this.we use Co-relation Coefficent(It is Objective)
np.corrcoef(df.yexp,df.sal)

#Import Liabrary
import statsmodels.formula.api as smf

#Simple Linear Regression
model=smf.ols('sal ~ yexp',data=df).fit()
model.summary()

#Predictions
pred=model.predict(pd.DataFrame(df['yexp']))

#Error Calculations(RMSE)
error=df.sal - pred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#Firstly we do log Transformations


model1=smf.ols('sal ~ np.log(yexp)',data=df).fit()
model1.summary()

#Predictions
pred1=model1.predict(pd.DataFrame(df['yexp']))

#Error Calculations(RMSE)
error=df.sal - pred1
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred1,'r')
plt.legend(['predicted line','observed data'])
plt.show()

#Secondly we do Exponential Transformations


model2=smf.ols('np.log(sal) ~ yexp',data=df).fit()
model2.summary()

#Predictions
pred2=model2.predict(pd.DataFrame(df['yexp']))
pred2_exp=np.exp(pred2)
pred2_exp
#Error Calculations(RMSE)
error=df.sal - pred2_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred2,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#Then Square Transformations


model3=smf.ols('sal ~ (yexp*yexp)',data=df).fit()
model3.summary()

#Predictions
pred3=model3.predict(pd.DataFrame(df['yexp']))

#Error Calculations(RMSE)
error=df.sal - pred3
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred3,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#then do SQRT transformations


model4=smf.ols('sal ~ np.sqrt(yexp)',data=df).fit()
model4.summary()

#Predictions
pred4=model4.predict(pd.DataFrame(df['yexp']))

#Error Calculations(RMSE)
error=df.sal - pred4
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred4,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#Lastly Polynominal Transformations
model5=smf.ols('np.log(sal) ~ yexp + I(yexp*yexp)',data=df).fit()
model5.summary()

#Predictions
pred5=model5.predict(pd.DataFrame(df['yexp']))
pred5_exp=np.exp(pred5)
pred5_exp
#Error Calculations(RMSE)
error=df.sal - pred_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred_exp,'r')
plt.legend(['predicted line','observed data'])
plt.show()




#The Best Model
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2)

finalmodel=smf.ols('sal ~ yexp',data=train).fit()
finalmodel.summary()

#Predictions On Test Data
testpred=finalmodel.predict(pd.DataFrame(test))

#RMSE on Test Data
error=test.sal - testpred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Predictions On Train Data
trainpred=finalmodel.predict(pd.DataFrame(train))

#RMSE on Test Data
error=train.sal - trainpred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse


