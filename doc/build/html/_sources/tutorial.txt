.. _tutorial:

Quick start tutorial
====================
**adenine** may be installed using standard Python tools (with
administrative or sudo permissions on GNU-Linux platforms)::

    $ pip install adenine

    or

    $ easy_install adenine

Installation from sources
-------------------------
If you like to manually install **adenine**, download the source tar.gz
from our
`BitBucket repository <https://bitbucket.org/samuele_fiorini/adenine>`_.
Then extract it and move into the root directory::

    $ tar xvf adenine-|release|.tar.gz
    $ cd adenine-|release|/

From here, you may use the standard Python installation step::

    $ python setup.py install

After **adenine** installation, you should have access to two scripts,
named with a common ``ade_`` prefix::

    $ ade_<TAB>
    ade_analysis.py    ade_run.py

This tutorial assumes that you downloaded and extracted **adenine**
source package which contains a ``examples`` directory with some ``.npy`` files
which will be used to show **adenine**'s' functionalities.

**adenine** needs only 3 ingredients:

* A ``X.npy`` input matrix
* A ``y.npy`` output vector (optional)
* A ``configuration`` file


Input data format
-----------------
Input data are assumed to be ``numpy`` array dumped in a ``.npy`` files organized with a row for each sample and a column for each feature.

.. _configuration:

Configuration File
------------------
**adenine** configuration file is a standard Python script. It is
imported as a module, then all the code is executed. In this file the user can define all the option needed to read the data and to create the pipelines.

.. literalinclude:: ../../adenine/examples/ade_config.py
   :language: python

.. _experiment:

Experiment runner
-----------------
The ``ade_run.py`` script, executes the full **adenine** framework. The prototype is the following::

    $ ade_run.py ade_config.py

When launched, the script reads the data, then it creates and runs each pipeline saving the results in a three-like structure which has the current folder as root.

.. _analysis:

Results analysis
----------------
This is the last step, needed to be performed in order to get some useful
summaries and plots from an already executed experiment.
The ``ade_analysis.py`` script accepts as only parameter a result directory
already created::

    $ ade_analysis.py result-dir

The script prints some results and produces a set of textual and graphical
results. An example of possible output obtained by one of the implemented pipelines is represented below.

.. figure:: _static/KMeans.png
   :align: center
