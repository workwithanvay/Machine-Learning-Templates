#Simpli linear regression

#Data preproccessing
#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


#splitting into test and training dataset
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#fitting the linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the results
y_pred=regressor.predict(x_test)

#visulaising the training set
plt.scatter(x_train, y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("salary vs years of experience(training)")
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set
plt.scatter(x_test, y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Years of Experience")
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()