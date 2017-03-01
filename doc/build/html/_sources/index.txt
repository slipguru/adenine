.. adenine documentation master file, created by
   sphinx-quickstart on Fri May 22 12:31:54 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
ADENINE (A Data ExploratioN pIpeliNE)
=====================================

**adenine** is a machine learning and data mining Python library for exploratory data analysis.

The main structure of **adenine** can be summarized in the following 4 steps.

1. **Imputing:** Does your dataset have missing entries? In the first step you can fill the missing values choosing between different strategies: feature-wise median, mean and most frequent value or k-NN imputing.

2. **Preprocessing:** Have you ever wondered what would have changed if only your data have been preprocessed in a different way? Or is it data preprocessing a good idea after all? **adenine** includes several preprocessing procedures, such as: data recentering, Min-Max scaling, standardization and normalization. **adenine** also allows you to compare the results of the analysis made with different preprocessing strategies.

3. **Dimensionality Reduction:** In the context of data exploration, this phase becomes particularly helpful for high dimensional data. This step includes manifold learning (such as isomap, multidimensional scaling, etc) and unsupervised feature learning (principal component analysis, kernel PCA, Bernoulli RBM, etc) techniques.

4. **Clustering:** This step aims at grouping data into clusters in an unsupervised manner. Several techniques such as k-means, spectral or hierarchical clustering are offered.

The final output of **adenine** is a compact, textual and graphical representation of the results obtained from the pipelines made with each possible combination of the algorithms selected at each step.

**adenine** can run on multiple cores/machines* and it is fully `scikit-learn` compliant.

User documentation
==================
.. toctree::
   :maxdepth: 2

 tutorial.rst

.. _api:

***********************
API
***********************

.. toctree::
   :maxdepth: 1


Pipeline utilities
-----------------------------

.. automodule:: adenine.core.define_pipeline
   :members:

.. automodule:: adenine.core.pipelines
   :members:

.. automodule:: adenine.core.analyze_results
   :members:

Input Data
-----------------------------

.. automodule:: adenine.utils.data_source
   :members:
   
   
Plotting functions
-----------------------------

.. automodule:: adenine.core.plotting
   :members:


Extra tools
-----------------------------

.. automodule:: adenine.utils.extra
   :members:


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

