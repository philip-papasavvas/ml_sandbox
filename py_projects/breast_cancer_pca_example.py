"""
Principal Component Analysis (PCA) on the scikit-learn breast cancer dataset.

This script illustrates two properties of PCA:

1. With no dimensionality reduction (n_components == n_features) the inverse
   transform reconstructs the original (scaled) data almost exactly.
2. As fewer principal components are retained, more variance is discarded and
   the reconstruction error grows.

Features are standardised first because PCA is sensitive to the scale of the
input variables.
"""
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # Load the dataset as a DataFrame so columns keep their feature names.
    cancer_data = datasets.load_breast_cancer()
    cancer_df = pd.DataFrame(cancer_data["data"], columns=cancer_data["feature_names"])
    print(f"Shape of data: {cancer_df.shape}")
    print(f"Dataset has {cancer_df.shape[1]} features\n")

    # Standardise to zero mean and unit variance: the raw features span very
    # different scales, which would otherwise dominate the principal components.
    scaler = StandardScaler()
    cancer_scaled = scaler.fit_transform(cancer_df)

    # Compare a range of component counts, up to the full feature set.
    for iteration, num_pc in enumerate([5, 10, 15, 20, 25, 30], start=1):
        print("*" * 30)
        print(f"Iteration {iteration}: retaining {num_pc} principal components")

        pca = PCA(n_components=num_pc)
        transformed = pca.fit_transform(cancer_scaled)

        explained = 100 * np.sum(pca.explained_variance_ratio_)
        print(f"Explained variance: {explained:.2f}%")

        # Map the compressed representation back to the original feature space
        # and measure how far the reconstruction drifts from the input.
        reconstructed = pca.inverse_transform(transformed)
        max_feature_drift = np.max(np.abs(np.sum(reconstructed - cancer_scaled, axis=0)))
        print(f"Max reconstruction drift across features: {max_feature_drift:.2E}\n")


if __name__ == "__main__":
    main()
