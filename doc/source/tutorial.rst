.. _tutorial:

Quick start tutorial
====================
**ADENINE** may be installed using standard Python tools (with
administrative or sudo permissions on GNU-Linux platforms)::

    $ pip install adenine

    or

    $ easy_install adenine

Installation from sources
-------------------------
If you like to manually install **ADENINE**, download the .zip or .tar.gz archive
from `<http://slipguru.github.io/adenine/>`_. Then extract it and move into the root directory::

    $ unzip slipguru-adenine-|release|.zip
    $ cd adenine-|release|/

or::

    $ tar xvf slipguru-adenine-|release|.tar.gz
    $ cd adenine-|release|/

Otherwise you can clone our `GitHub repository <https://github.com/slipguru/adenine>`_::

   $ git clone https://github.com/slipguru/adenine.git

From here, you can follow the standard Python installation step::

    $ python setup.py install

After **ADENINE** installation, you should have access to two scripts,
named with a common ``ade_`` prefix::

    $ ade_<TAB>
    ade_analysis.py    ade_run.py

This tutorial assumes that you downloaded and extracted **ADENINE**
source package which contains a ``examples`` directory with some ``.npy`` files
which will be used to show **ADENINE** functionalities.

**ADENINE** needs only 3 ingredients:

* A ``n_samples x n_variables`` input matrix
* A ``n_samples x 1`` output vector (optional)
* A ``configuration`` file


Input data format
-----------------
Input data are assumed to be:

* ``numpy`` array stored in ``.npy`` files organized with a row for each sample and a column for each feature,
* tabular data stored in comma separated ``.csv`` files presenting the variables header on the first row and the sample indexes on the first column,
* toy examples available from ``adenine.utils.data_source`` function.

.. _configuration:

Configuration File
------------------
**ADENINE** configuration file is a standard Python script. It is
imported as a module, then all the code is executed. In this file the user can define all the option needed to read the data and to create the pipelines.

.. literalinclude:: ../../adenine/ade_config.py
   :language: python

.. _experiment:

Experiment runner
-----------------
The ``ade_run.py`` script, executes the full **ADENINE** framework. The prototype is the following::

    $ ade_run.py ade_config.py

When launched, the script reads the data, then it creates and runs each pipeline saving the results in a three-like structure which has the current folder as root.

.. _analysis:

Results analysis
----------------
The ``ade_analysis.py`` script provides useful summaries and graphs from the results of the experiment. This script accepts as only parameter a result directory
already created::

    $ ade_analysis.py result-dir

The script produces a set of textual and graphical results. An output example obtained by one of the implemented pipelines is represented below.

.. image:: pca.png
   :scale: 80 %
   :alt: broken link

.. image:: kpca.png
   :scale: 80 %
   :alt: broken link

You can reproduce the example above specifying ``data_source.load('circles')`` in the configuration file.

Example dataset
----------------
An example dataset can be dowloaded :download:`here <TCGA-PANCAN-HiSeq-801x20531.tar.gz>`. The dataset is a random extraction of 801 samples (with dimension 20531) measuring RNA-Seq gene expression of patients affected by 5 different types of tumor: breast invasive carcinoma (BRCA), kidney renal clear cell carcinoma (KIRC), colon  (COAD), lung  (LUAD) and prostate adenocarcinoma (PRAD). The full dataset is maintained by The Cancer Genome Atlas Pan-Cancer Project [1] and we refer to the `original repository <https://www.synapse.org/#!Synapse:syn4301332>`_ for furher details.

Reference
----------------
[1] Weinstein, John N., et al. "The cancer genome atlas pan-cancer analysis project." Nature genetics 45.10 (2013): 1113-1120.

