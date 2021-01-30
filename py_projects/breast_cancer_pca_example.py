"""
Created on 31 Jan 2021

Example to show myself that PCA when you don't drop any
dimensions gives you back the same data with the transform,
and show the amount of variance lost with different num
of principal components
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# load the data
cancer_data = datasets.load_breast_cancer()
cancer = cancer_data['data']

print(f"Shape of data is: {cancer.shape}")

cancer_df = pd.DataFrame(cancer,
                         columns=cancer_data['feature_names'])
cancer_df.head(3)

# carry out feature scaling
# variables are not similar scales
scaler = StandardScaler()
# fit the scaler
scaler.fit(cancer_df)
# normalise the data - 0 mean, standard deviation 1
cancer_scaled = scaler.transform(cancer)

# use describe to confirm the mean and st dev
pd.DataFrame(cancer_scaled,
             columns=cancer_data['feature_names']).describe()

print(f'Cancer dataset has {len(cancer_data.feature_names)} different features')

# try a few different iterations of PCA using different number of
# components
num_pc_iter = [5, 10, 15, 20, 25, 30]
for i, num_pc in enumerate(num_pc_iter, 1): # get enumerate to start at one
    # use PCA on the transformed dataset to reduce dimensions
    print("*"*30)
    print(f"Iteration {i}. \n "
          f"Number of principal components to choose: {num_pc}")
    pca = PCA(n_components=num_pc)
    pca.fit(cancer_scaled)
    # transform the data using the num of principal components
    transformed_data = pca.transform(cancer_scaled)
    transformed_variance = pca.explained_variance_ratio_

    # print(f"The original shape of data was {cancer.shape}. \n"
    #       f"The shape of the new mapped data is {transformed_data.shape} \n")

    print(f"Explained {100 * np.cumsum(transformed_variance)[-1]: .2f}% of total variance ")

    # map the data back to original space following compression
    cancer_re_mapped = pca.inverse_transform(transformed_data)
    diff_new_old = cancer_re_mapped - cancer_scaled

    print(f"Max sum of differences for features (between newly mapped and original)"
          f" {np.max(np.sum(diff_new_old, axis=0)) :.2E} \n")
