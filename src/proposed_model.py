import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

from .feature_engineering import DualPerspectiveFeatureRefiner
from .attention_layer import MultiHeadFeatureAttentionGate
from .pso_optimizer import SwarmEnsembleOptimizer

class HGFASNet(BaseEstimator, ClassifierMixin):
    """
    HG-FASNet: Hybrid Gated Feature-Attention Stacking Network with Metaheuristic Swarm Optimization.
    
    A novel, multi-tier architectural framework designed to overcome the linearity
    and rigid weighting limitations of previous clinical decision models (e.g., HRFLM).
    
    System Architecture:
    -------------------
    Tier 1: Dual-Perspective Feature Refinement (DPFR)
            - Clinically validated physiological non-linear interaction synthesis
            - Outlier-resistant RobustScaling
    Tier 2: Multi-Head Cross-Feature Attention Gate (MHC-FAG)
            - Scaled dot-product cross-attribute self-attention
            - Adaptive data-dependent sigmoid gating vector
    Tier 3: Tri-Architectural Heterogeneous Base Learners (TA-HBL)
            - Leaf-wise Gradient Boosting (LightGBM)
            - Bagged Decorrelated Ensembles (Random Forest)
            - Calibrated Radial Basis Kernel Geometric Classifier (Calibrated RBF-SVM)
    Tier 4: Metaheuristic Particle Swarm Optimization (PSO-MCE)
            - Out-of-fold probabilistic validation to prevent data leakage
            - Global swarm optimization of hypothesis blending weights w* and threshold tau*
    """
    
    def __init__(
        self,
        n_pso_particles=50,
        n_pso_iter=80,
        cv_folds=5,
        random_state=42
    ):
        self.n_pso_particles = n_pso_particles
        self.n_pso_iter = n_pso_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Pipelines
        self.dpfr = DualPerspectiveFeatureRefiner(seed=random_state)
        self.attention_gate = None
        self.pso_optimizer = SwarmEnsembleOptimizer(
            n_particles=n_pso_particles,
            n_iter=n_pso_iter,
            seed=random_state
        )
        
        # Base learners
        self.model_lgb = None
        self.model_rf = None
        self.model_svm = None

    def _init_base_models(self):
        return {
            "lgbm": lgb.LGBMClassifier(
                n_estimators=80,
                learning_rate=0.03,
                max_depth=3,
                num_leaves=7,
                random_state=self.random_state,
                verbose=-1
            ),
            "rf": RandomForestClassifier(
                n_estimators=120,
                max_depth=5,
                random_state=self.random_state
            ),
            "svm": CalibratedClassifierCV(
                estimator=SVC(C=1.2, kernel="rbf", random_state=self.random_state),
                cv=3
            )
        }

    def fit(self, X, y):
        # Tier 1: Dual-Perspective Feature Refinement
        X_dpfr = self.dpfr.fit_transform(X, y)
        
        # Tier 2: Multi-Head Feature-Attention Gate
        d_feat = X_dpfr.shape[1]
        self.attention_gate = MultiHeadFeatureAttentionGate(d_features=d_feat, seed=self.random_state)
        X_trans = self.attention_gate.fit_transform(X_dpfr)
        
        # Tier 3 & 4: Out-of-Fold Probability Generation for PSO Meta-Learning
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        oof_probs = np.zeros((len(X), 3))
        
        for tr_idx, val_idx in skf.split(X_trans, y):
            fold_models = self._init_base_models()
            fold_models["lgbm"].fit(X_trans[tr_idx], y[tr_idx])
            fold_models["rf"].fit(X_trans[tr_idx], y[tr_idx])
            fold_models["svm"].fit(X_trans[tr_idx], y[tr_idx])
            
            oof_probs[val_idx, 0] = fold_models["lgbm"].predict_proba(X_trans[val_idx])[:, 1]
            oof_probs[val_idx, 1] = fold_models["rf"].predict_proba(X_trans[val_idx])[:, 1]
            oof_probs[val_idx, 2] = fold_models["svm"].predict_proba(X_trans[val_idx])[:, 1]
            
        # Optimize ensemble weights and cutoff threshold with PSO
        self.pso_optimizer.fit(oof_probs, y)
        
        # Fit final full base models on all transformed training data
        full_models = self._init_base_models()
        self.model_lgb = full_models["lgbm"].fit(X_trans, y)
        self.model_rf = full_models["rf"].fit(X_trans, y)
        self.model_svm = full_models["svm"].fit(X_trans, y)
        
        return self

    def _get_base_probabilities(self, X):
        X_dpfr = self.dpfr.transform(X)
        X_trans = self.attention_gate.transform(X_dpfr)
        
        p_lgb = self.model_lgb.predict_proba(X_trans)[:, 1]
        p_rf = self.model_rf.predict_proba(X_trans)[:, 1]
        p_svm = self.model_svm.predict_proba(X_trans)[:, 1]
        
        return np.column_stack([p_lgb, p_rf, p_svm])

    def predict_proba(self, X):
        base_probs = self._get_base_probabilities(X)
        final_p1 = self.pso_optimizer.predict_proba(base_probs)
        return np.column_stack([1.0 - final_p1, final_p1])

    def predict(self, X):
        base_probs = self._get_base_probabilities(X)
        return self.pso_optimizer.predict(base_probs)

    @property
    def optimal_weights(self):
        return self.pso_optimizer.best_weights_

    @property
    def optimal_threshold(self):
        return self.pso_optimizer.best_threshold_
