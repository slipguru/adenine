<p align="center">
  <img src="http://www.slipguru.unige.it/Software/adenine/_static/ade_logo_bitmap.png"><br><br>
</p>
-----------------

# Adenine: A data exploration pipeline

**adenine** is a machine learning and data mining Python library for exploratory data analysis.

The main structure of **adenine** can be summarized in the following 4 steps.

1. **Imputing:** Does your dataset have missing entries? In the first step you can fill the missing values choosing between different strategies: feature-wise median, mean and most frequent value or k-NN imputing.

2. **Preprocessing:** Have you ever wondered what would have changed if only your data have been preprocessed in a different way? Or is it data preprocessing a good idea after all? **adenine** includes several preprocessing procedures, such as: data recentering, Min-Max scaling, standardization and normalization. **adenine** also allows you to compare the results of the analysis made with different preprocessing strategies.

3. **Dimensionality Reduction:** In the context of data exploration, this phase becomes particularly helpful for high dimensional data. This step includes manifold learning (such as isomap, multidimensional scaling, etc) and unsupervised feature learning (principal component analysis, kernel PCA, etc) techniques.

4. **Clustering:** This step aims at grouping data into clusters in an unsupervised manner. Several techniques such as k-means, spectral or hierarchical clustering are offered.

The final output of **adenine** is a compact, textual and graphical representation of the results obtained from the pipelines made with each possible combination of the algorithms selected at each step.

**adenine** can run on multiple cores/machines* and it is fully `scikit-learn` compliant.

## Installation

**adenine** supports Python 2.7.

### Pip installation
`$ pip install adenine`

### Installing from sources
```bash
$ git clone https://github.com/slipguru/adenine
$ cd adenine
$ python setup.py install
```

## Try Adenine

### 1. Create your configuration file
Start from the provided template and edit your configuration file with your favourite text editor
```bash
$ ade_run.py -c my-config-file.py
$ vim my-config-file.py
...
```
```python
from adenine.utils import data_source

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = '_experiment'
output_root_folder = 'results'
plotting_context = 'notebook'  # one of {paper, notebook, talk, poster}
file_format = 'pdf'  # or 'png'

# ----------------------------  INPUT DATA ---------------------------- #
# Load an example dataset or specify your input data in tabular format
X, y, feat_names, index = data_source.load('iris')

# -----------------------  PIPELINES DEFINITION ------------------------ #
# --- Missing Values Imputing --- #
step0 = {'Impute': [True, {'missing_values': 'NaN',
                            'strategy': ['nearest_neighbors']}]}

# --- Data Preprocessing --- #
step1 = {'MinMax': [True, {'feature_range': [(0, 1)]}]}

# --- Unsupervised feature learning --- #
step2 = {'KernelPCA': [True, {'kernel': ['linear', 'rbf', 'poly']}],
         'Isomap': [False, {'n_neighbors': 5}],
         'MDS': [True, {'metric': True}],
         'tSNE': [False],
         'RBM': [True, {'n_components': 256}]
         }

# --- Clustering --- #
# affinity ca be precumputed for AP, Spectral and Hierarchical
step3 = {'KMeans': [True, {'n_clusters': [3, 'auto']}],
         'Spectral': [False, {'n_clusters': [3]}],
         'Hierarchical': [False, {'n_clusters': [3],
                                  'affinity': ['euclidean'],
                                  'linkage':  ['ward', 'average']}]
         }
```

### 2. Run the pipelines
```bash
$ ade_run.py my-config-file.py
```

### 3. Automatically generate beautiful publication-ready plots and textual results
```bash
$ ade_analysis.py results/ade_experiment_<TODAY>
```

## *Got large datasets?

**adenine** takes advantage of `mpi4py` to distribute the execution of the pipelines on HPC architectures
```bash
$ mpirun -np <MPI-TASKS> --hosts <HOSTS-LIST> ade_run.py my-config-file.py
```
