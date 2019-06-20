import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
data_set = pd.read_csv('cc.csv')
data_set = data_set.apply(LabelEncoder().fit_transform)
nulls = pd.DataFrame(data_set.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
print(nulls)
data_set['MINIMUM_PAYMENTS'].fillna((data_set['MINIMUM_PAYMENTS'].mean()), inplace=True)
print(data_set["MINIMUM_PAYMENTS"].isnull().any())
data_set['CREDIT_LIMIT'].fillna((data_set['CREDIT_LIMIT'].mean()), inplace=True)
print(data_set["CREDIT_LIMIT"].isnull().any())

x = data_set.iloc[:, [0, 1, 2, 3, 4, 5, 12, 13]]

print(x)
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns=x.columns)
nclusters = 3
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)
y_cluster = km.predict(X_scaled)

print(y_cluster)
wcss= []# elbow method to know the number of clusters
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 15), wcss)
plt.title('the elbow method')

plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()



