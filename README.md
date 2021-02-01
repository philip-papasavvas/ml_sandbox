# ml_sandbox
This repository contains all things machine learning with the following packages:
* notebook-projects
    * Breast Cancer - PCA
        - Principal component analysis on **breast_cancer** dataset from [scikit-learn](https://scikit-learn.org/stable/)
        - Demonstrated difference in amount of variation retained using different choices for number of principal components
    * Natural Language Processing - SMS spam dataset
        - Dataset from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset), and is also found on the **SMS spam** dataset from [UCI Machine learning](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
        - Submitted on Kaggle [here](https://www.kaggle.com/philpap/logisticregression-decisiontree-neuralnetwork)
        - Exploratory data analysis, feature engineering (attributes of the messages, such as message length, word count, etc.)
        - Used tokenisation, vectorisation, and bag-of-words techniques
        - Applied three supervised learning methods to data: 
            * logistic regression
            * sequential neural network using [Keras](https://keras.io/)
            * decision tree algorithm
         Achieved test set accuracy of over 95% for all models implemented      
* pyTorch tutorials (using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) workbooks for GPU), taken from the [60 minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
  * Autograd - automatic differentiation for all operations on tensors
  * CIFAR10 - 60,000 images consisting of images belonging to 10 different classes, (airplane, automobile, bird, etc.), taken from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
  * Neural Networks - walk through of how to use pyTorch
  * Tensors introduction
