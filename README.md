=====================================
ADENINE (A Data ExploratioN pIpeliNE)
=====================================

**ADENINE** is a machine learning and data mining Python pipeline that helps you to answer this tedious question: are my data relevant with the problem I'm dealing with?

The main structure of adenine can be summarized in the following 4 steps.

1. **Imputing:** Does your dataset have missing entries? In the first step you can fill the missing values choosing between different strategies: feature-wise median, mean and most frequent value or a more stable k-NN imputing.

2. **Preprocessing:** Have you ever wondered what would have changed if only  your data have been preprocessed in a different way? Or is data preprocessing is a good idea at all? ADENINE offers several preprocessing procedures, such as: data centering, Min-Max scaling, standardization or normalization and allows you to compare the results of the analysis made with different preprocessing step as starting point.

3. **Dimensionality Reduction:** In the context of data exploration, this  phase becomes particularly helpful for high dimensional data (e.g. -omics scenario). This step includes some manifold learning (such as isomap, multidimensional scaling, etc) and unsupervised dimensionality reduction (principal component analysis, kernel PCA) techniques.

4. **Clustering:** This step aims at grouping data into clusters in an unsupervised manner. Several techniques such as k-means, spectral or hierarchical clustering are offered.

The final output of adenine is a compact and textual representation of the results obtained from the pipelines made with each possible combination of the algorithms implemented at each step.
