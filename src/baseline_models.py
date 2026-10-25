import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class HRFLM_BasePaper:
    """
    Replication of the Base Paper:
    Mohan, S., Thirumalai, C., & Srivastava, G. (2019).
    'Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques.'
    IEEE Access, 7, 81542-81554.
    
    Architecture:
    Hybrid Random Forest with Linear Model (HRFLM).
    1. Computes feature importance scores using Random Forest (n_estimators=100).
    2. Filters the feature space down to the most significant clinical subset (top 9 features).
    3. Trains a Generalized Linear Model (Logistic Regression with L2 regularization).
    """
    
    def __init__(self, n_top_features=9, random_state=42):
        self.n_top_features = n_top_features
        self.random_state = random_state
        self.rf_selector = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.scaler = StandardScaler()
        self.glm = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        self.selected_indices_ = None

    def fit(self, X, y):
        # Step 1: Feature weighting via Random Forest
        self.rf_selector.fit(X, y)
        importances = self.rf_selector.feature_importances_
        self.selected_indices_ = np.argsort(importances)[::-1][:self.n_top_features]
        
        # Step 2: Linear Model on selected subset
        X_sub = X[:, self.selected_indices_]
        X_scaled = self.scaler.fit_transform(X_sub)
        self.glm.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_sub = X[:, self.selected_indices_]
        X_scaled = self.scaler.transform(X_sub)
        return self.glm.predict(X_scaled)

    def predict_proba(self, X):
        X_sub = X[:, self.selected_indices_]
        X_scaled = self.scaler.transform(X_sub)
        return self.glm.predict_proba(X_scaled)

def get_standard_baselines(seed=42):
    """
    Returns a dictionary of standard baseline models with appropriate scaling.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "Support Vector Machine (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=seed))
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree (CART)": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "Base Paper HRFLM (Mohan et al., 2019)": HRFLM_BasePaper(n_top_features=9, random_state=seed)
    }
