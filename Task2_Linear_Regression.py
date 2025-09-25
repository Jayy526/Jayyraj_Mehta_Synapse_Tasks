import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('insurance.csv')

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

X = df.iloc[:,0:6]
y = df['charges']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=45,test_size=0.3)
print(X_train.shape)
model = LinearRegression()
model.fit(X_train,y_train)
#prediction
y_pred = model.predict(X_test)
print(y_pred.shape)
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)



plt.plot()
plt.scatter(y_test,y_pred,alpha=0.6, color='blue', label="Predictions")
plt.plot(y_test, p(y_test), color='red', label="Best Fit Line")
plt.legend()
plt.show()
