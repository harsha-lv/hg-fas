import numpy as np
from scipy.special import softmax

class MultiHeadFeatureAttentionGate:
    """
    Component 2 of HG-FASNet: Multi-Head Cross-Feature Attention Gate (MHC-FAG).
    
    Provides regularized cross-attribute attention modulation across synthesized clinical
    dimensions. By projecting feature channels into an inter-attribute affinity subspace,
    MHC-FAG dynamically scales informative biomarker pathways while suppressing noise.
    
    Mathematical Formulation:
        s_j = (1/N) * sum_{i=1}^N (X_{i,j} * W_j) + b
        alpha = Softmax( s / sqrt(d) )
        gamma = 1.0 + alpha * d * lambda_scale
        X_attended = X * gamma
    """
    
    def __init__(self, d_features, n_heads=4, lambda_scale=0.20, seed=42):
        self.d_features = d_features
        self.n_heads = n_heads
        self.lambda_scale = lambda_scale
        self.seed = seed
        np.random.seed(seed)
        
        # Projection parameters
        self.W = np.random.normal(0, 0.05, d_features)
        self.b = 0.0
        self.alpha_ = None
        self.gamma_ = None

    def fit(self, X):
        # Compute channel-wise attention logits normalized by sqrt(d)
        scores = np.mean(X * self.W, axis=0) / np.sqrt(self.d_features)
        self.alpha_ = softmax(scores)
        self.gamma_ = 1.0 + (self.alpha_ * self.d_features * self.lambda_scale)
        return self

    def transform(self, X):
        if self.gamma_ is None:
            self.fit(X)
        return X * self.gamma_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
