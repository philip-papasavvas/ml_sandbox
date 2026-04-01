# ml_sandbox
A repository for machine learning projects, tutorials, and reference material.

## notebook-projects
* **Breast Cancer - PCA**
    - Principal component analysis on the **breast_cancer** dataset from [scikit-learn](https://scikit-learn.org/stable/)
    - Demonstrated difference in amount of variation retained using different choices for number of principal components
* **Natural Language Processing - SMS spam classifier**
    - Dataset from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset), also found on [UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
    - Submitted on Kaggle [here](https://www.kaggle.com/philpap/logisticregression-decisiontree-neuralnetwork)
    - Exploratory data analysis, feature engineering (message length, word count, etc.)
    - Tokenisation, vectorisation, and bag-of-words techniques
    - Applied three supervised learning methods:
        * logistic regression
        * sequential neural network using [Keras](https://keras.io/)
        * decision tree algorithm
    - Achieved test set accuracy of over 95% for all models
* **Credit Card Churn Modelling**
    - Churn prediction on the BankChurners dataset
* **Diabetes - Linear Regression**
    - Linear regression applied to the diabetes dataset
* **Whoop - Exploratory Data Analysis**
    - EDA on WHOOP fitness tracker data, focusing on sleep metrics
* **HYROX Simulation**
    - Monte Carlo-style simulation of a [HYROX](https://hyrox.com/en/) fitness race
    - Modelled each station and run leg using normal/uniform distributions to estimate finish times

## py_projects
Standalone Python scripts:
* **breast_cancer_pca_example.py** - PCA on the breast cancer dataset, comparing original vs reconstructed data
* **numba_example.py** - Demonstrates [Numba](https://numba.pydata.org/) JIT compilation and benchmarks the performance gain
* **whoop_source_eda.py** - EDA on WHOOP sleep data with feature importance analysis using linear regression

## pytorch_tutorials
[PyTorch](https://pytorch.org/) tutorials (using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) for GPU), taken from the [60 minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html):
* Tensors introduction
* Autograd - automatic differentiation for all operations on tensors
* Neural Networks - walkthrough of building neural networks in PyTorch
* CIFAR-10 - image classification on 60,000 images across 10 classes, from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* Data loading

## Other notebooks
* **coursera_machine_learning_wisdom_nuggets.ipynb** - Key takeaways from Andrew Ng's [Coursera Machine Learning course](https://www.coursera.org/learn/machine-learning) (gradient descent, classification, overfitting/regularisation)
* **data_science_interview_questions.ipynb** - Reference notes on common data science and statistics interview topics
