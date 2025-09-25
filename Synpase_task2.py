import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('insurance.csv')
plt.plot()
plt.hist(df['age'],edgecolor = "black")
plt.title("age distribution")
plt.show()
plt.plot()
plt.hist(df['bmi'],edgecolor = "white")
plt.title("bmi distribution")
plt.show()
#plot scatterplots for no. of children and charges, and also for bmi and charges
plt.plot()
plt.scatter(df['children'],df['charges'])
plt.scatter(df['bmi'],df['charges'],color = "black")
plt.show()
#use boxplots to find correlation between Sex and Charges, between Smoker and charges, and also between region and charges
sb.boxplot(data= df,x = df['sex'],y = df['charges'],color="skyblue")
plt.show()
sb.boxplot(data= df,x = df['smoker'],y = df['charges'],color="yellow")
plt.show()
#Now, make a heatmap to understand correlation between all attributes
corr = df.corr(numeric_only=True)
sb.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.show()
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
print(df.head())