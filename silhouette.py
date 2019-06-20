import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
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
x = data_set.iloc[:, [9, 10]]
print(x)
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns=x.columns)
nclusters = 2
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)
y_cluster = km.predict(X_scaled)
print(y_cluster)
score = metrics.silhouette_score(X_scaled, y_cluster)
print('silhouette_score :', score)


