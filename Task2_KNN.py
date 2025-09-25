import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("classified_data.txt", index_col=0)
print(df.head())
X = df.iloc[:,0:10]
y = df.iloc[:,10]
features = X.columns
for i, feature in enumerate(features, 1):
    plt.subplot(4, 3, i)  # adjust rows/cols as needed
    sb.boxplot(x=df['TARGET CLASS'], y=df[feature], palette="Set2")
    plt.title(f"Boxplot of {feature} by TARGET CLASS")
    plt.xlabel("TARGET CLASS")
    plt.ylabel(feature)
plt.show()


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
df_feat = pd.DataFrame(X_scale,columns=df.columns[:-1])
print(df_feat.head())
X_train,X_test,y_train,y_test = train_test_split(X_scale,y,random_state=45,test_size=0.3)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
print(np.round(accuracy_score(y_test,y_pred),3))
accuracies = []
for k in range (1,61):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    y_pred= model.predict(X_test)
    accuracies.append(accuracy_score(y_test,y_pred))

plt.figure(figsize=(10,6))
plt.plot(range(1, 61), accuracies, marker='o', linestyle='-')
plt.title("K Value vs Accuracy")
plt.xlabel("K (n_neighbors)")
plt.ylabel("Accuracy")
plt.xticks(range(0, 61, 5))
plt.grid(True)
plt.show()
print("Best value of k is 6")
