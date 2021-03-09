import numpy as np 

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    
    def fit(self, X):
        
        # Calculate the mean of `X`
        self.mean = np.mean(X, axis=0)

        # Centering of Mean for variance
        X = X - self.mean

        # Calculate the Covariances from given `X` functions needs samples as columns
        cov = np.cov(X.T)

        # Calculate the eigenvalues and eigenvectors 
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort the eigenvectors in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store the first n eigenvectors inside `self.components`
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


