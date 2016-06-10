from adenine.utils import data_source
X,y,feat_names,class_names = data_source.load("custom","data.csv","labels.csv")
step1 = {'Normalize':[True,{'norm':'l2'}]} #Preprocessing
step2 = {'PCA':[True,{'n_components':2}],'KernelPCA':[True,{'kernel': ['rbf'], \
'n_components':3,'gamma':2}],'Isomap':[True,{'n_components':3}]} #Dim. Reduction
step3 = {'KMeans':[True,{'n_clusters':['auto']}]} #Clustering
