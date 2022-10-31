import inline as inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Reading Data
file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
# Data Wrangling
pd.set_option('expand_frame_repr', None)
df.drop(['Unnamed: 0','id'],axis=1,inplace=True)
print(df.head(5))
print(df.dtypes)
print(df.describe(include='all'))
print(df.isnull().sum())
df['bedrooms'].replace(np.nan,df['bedrooms'].mean(),inplace=True)
df['bathrooms'].replace(np.nan,df['bathrooms'].mean(),inplace=True)
print(df.isnull().sum())
# Exploratory Data Analysis
print(df['floors'].value_counts().to_frame())
print(df['waterfront'].value_counts().to_frame())
print(df['view'].value_counts().to_frame())
print(df.corr()['price'].sort_values())
sns.boxplot(x=df['waterfront'],y=df['price'])
plt.show()
sns.regplot(x=df['sqft_above'],y=df['price'])
plt.show()
# Model Development
 # Trial on long variable with low correlation with price
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
y_predict=lm.predict(X)
print(y_predict)
print(lm.score(X, Y))
print(lm.coef_)
# Building a regression Model to predict house prices based on the area
X_sqt=df[['sqft_above']]
lm = LinearRegression()
lm.fit(X_sqt,Y)
y_predict_sqt=lm.predict(X_sqt)
print(y_predict_sqt)
print(lm.score(X_sqt,Y))
# Building a regression Model to predict house prices based on multiple variables
z=df[["floors", "waterfront","lat" ,"bedrooms","sqft_basement","view","bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
lm.fit(z,Y)
y_predict_z=lm.predict(z)
print(y_predict_z)
print(lm.score(z,Y))
print(lm.coef_)
# Creating a pipeline to predict the price
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
Pipe=Pipeline(Input)
Pipe.fit(z,Y)
print(Pipe.predict(z))
print(Pipe.score(z,Y))
# Model Evaluation
x_train, x_test, y_train, y_test = train_test_split(z, Y, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape)
print("number of training samples:",x_train.shape)
print("test samples:", x_test.shape)
print("training samples:",x_train.shape)
scores=cross_val_score(Pipe,z,Y,cv=5)
print(scores)
print(np.mean(scores))
# Model Refinement
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
Y_final=RidgeModel.predict(x_test)
print(Y_final)
print(RidgeModel.score(x_test,Y_final))
# Increasing the order of the polynomial function
pr=PolynomialFeatures(degree=2,include_bias=False)
poly=pr.fit_transform(z,Y)