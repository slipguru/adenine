from adenine.utils import data_source
X, y, feats, classes = data_source.load('custom', 'data.csv', 'labels.csv')
step1 = {'Normalize': [True, {'norm': 'l2'}]} # Preprocessing
step2 = {'KernelPCA': [True, {'kernel': ['rbf'], 'n_components': 3, 'gamma':
2}], 'Isomap': [True, {'n_components': 3}]} # Dimensionality reduction
step3 = {'KMeans': [True, {'n_clusters': ['auto']}]} # Clustering
