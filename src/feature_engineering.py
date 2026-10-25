import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

class DualPerspectiveFeatureRefiner:
    """
    Component 1 of HG-FASNet: Dual-Perspective Feature Refinement (DPFR).
    
    1. Synthesizes clinical biomarker non-linear interactions:
       - Rate Pressure Product (RPP = trestbps * thalach / 1000)
       - Cholesterol-Age Ratio (CAR = chol / (age + 1))
       - Cardiac Pathological Burden (CPB = 2.0*oldpeak + 1.5*ca + 1.0*thal)
       - Exercise Ischemia Burden (EIB = exang * (slope + 1))
       - Chest Pain to Heart Rate Index (CPM = cp / ((thalach / 100) + 1))
    2. Scales via RobustScaler (median-centered, IQR scaled to prevent outlier distortion).
    3. Analyzes feature relevance through dual perspective:
       - Tree-based non-linear split importance
       - Information-theoretic Mutual Information
    """
    
    ENGINEERED_FEATURE_NAMES = [
        "RPP_RatePressureProduct",
        "CAR_CholAgeRatio",
        "CPB_PathologicalBurden",
        "EIB_IschemiaBurden",
        "CPM_ChestPainRateIndex"
    ]
    
    def __init__(self, top_k=None, seed=42):
        self.top_k = top_k
        self.seed = seed
        self.scaler = RobustScaler()
        self.selected_indices_ = None
        self.feature_scores_ = None
        
    def _create_interactions(self, X):
        age = X[:, 0]
        sex = X[:, 1]
        cp = X[:, 2]
        trestbps = X[:, 3]
        chol = X[:, 4]
        fbs = X[:, 5]
        restecg = X[:, 6]
        thalach = X[:, 7]
        exang = X[:, 8]
        oldpeak = X[:, 9]
        slope = X[:, 10]
        ca = X[:, 11]
        thal = X[:, 12]
        
        # Clinically validated physiological interactions
        rpp = (trestbps * thalach) / 1000.0
        chol_age = chol / (age + 1.0)
        cardiac_risk = (oldpeak * 2.0) + (ca * 1.5) + (thal * 1.0)
        ischemia_burden = exang * (slope + 1.0)
        cp_thalach_ratio = cp / ((thalach / 100.0) + 1.0)
        
        return np.column_stack([X, rpp, chol_age, cardiac_risk, ischemia_burden, cp_thalach_ratio])

    def fit_transform(self, X, y):
        X_eng = self._create_interactions(X)
        X_scaled = self.scaler.fit_transform(X_eng)
        
        if self.top_k is not None and self.top_k < X_scaled.shape[1]:
            # Dual-perspective feature ranking
            rf = RandomForestClassifier(n_estimators=100, random_state=self.seed)
            rf.fit(X_scaled, y)
            gini_scores = rf.feature_importances_ / (np.max(rf.feature_importances_) + 1e-8)
            
            mi_scores = mutual_info_classif(X_scaled, y, random_state=self.seed)
            mi_scores = mi_scores / (np.max(mi_scores) + 1e-8)
            
            self.feature_scores_ = 0.5 * gini_scores + 0.5 * mi_scores
            self.selected_indices_ = np.argsort(self.feature_scores_)[::-1][:self.top_k]
            return X_scaled[:, self.selected_indices_]
            
        return X_scaled

    def transform(self, X):
        X_eng = self._create_interactions(X)
        X_scaled = self.scaler.transform(X_eng)
        if self.selected_indices_ is not None:
            return X_scaled[:, self.selected_indices_]
        return X_scaled
