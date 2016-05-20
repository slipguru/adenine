from adenine.utils import data_source
X, y, feats, classes = data_source.load('custom', 'data.csv', 'labels.csv')
step1 = {'None': [True]} # Preprocessing
step2 = {'KernelPCA': [True, {'n_components': 2,
'kernel': ['linear','rbf'], 'gamma': 2}]} # Dimensionality reduction
step3 = {'KMeans': [True, {'n_clusters': [2]}]} # Clustering
