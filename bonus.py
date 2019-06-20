import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

data_set = pd.read_csv('CC.csv')
data_set=data_set.apply(LabelEncoder().fit_transform)
nulls = pd.DataFrame(data_set.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns  = ['Null Count']
print(nulls)
data_set['CREDIT_LIMIT'].fillna((data_set['CREDIT_LIMIT'].mean()), inplace=True)
print(data_set["CREDIT_LIMIT"].isnull().any())
data_set['MINIMUM_PAYMENTS'].fillna((data_set['MINIMUM_PAYMENTS'].mean()), inplace=True)
print(data_set["MINIMUM_PAYMENTS"].isnull().any())


x = data_set.iloc[:, [9, 10]]
print(x)




scaler = preprocessing.StandardScaler()

scaler.fit(x)

X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


nclusters = 3 # this is the k in kmeans
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
print(y_cluster_kmeans)

wcss= []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()



ndimensions = 2

pca = PCA(ndimensions)



X_pca = pca.fit_transform(X_scaled)
nclusters = 3
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_pca)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_pca)
print(y_cluster_kmeans)
score = metrics.silhouette_score(X_pca, y_cluster_kmeans)
print("silhouette score after pca",score)