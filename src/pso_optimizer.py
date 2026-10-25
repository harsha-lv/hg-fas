import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SwarmEnsembleOptimizer:
    """
    Component 4 of HG-FASNet: Particle Swarm Optimization Metaheuristic Engine (PSO-MCE).
    
    Optimizes ensemble hypothesis blending weights w_m and clinical decision cutoff tau*
    using swarm intelligence to maximize harmonic diagnostic fitness on validation folds.
    
    Fitness Function:
        Fitness = 0.35 * F1 + 0.30 * Accuracy + 0.20 * Precision + 0.15 * Recall
    """
    
    def __init__(self, n_particles=50, n_iter=80, w_inertia=0.65, c1=1.5, c2=1.5, seed=42):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w_inertia = w_inertia
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.best_weights_ = None
        self.best_threshold_ = 0.50
        self.best_fitness_ = -float("inf")
        self.history_ = []

    def _fitness(self, particle, probs, y_true):
        n_models = probs.shape[1]
        raw_w = particle[:n_models]
        weights = raw_w / (np.sum(raw_w) + 1e-8)
        threshold = particle[n_models]
        
        ensemble_prob = np.dot(probs, weights)
        preds = (ensemble_prob >= threshold).astype(int)
        
        acc = accuracy_score(y_true, preds)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        return 0.35 * f1 + 0.30 * acc + 0.20 * prec + 0.15 * rec

    def fit(self, oof_probs, y_true):
        np.random.seed(self.seed)
        n_models = oof_probs.shape[1]
        dim = n_models + 1  # n_models weights + 1 threshold
        
        # Position initialization
        pos = np.random.uniform(0.1, 1.0, (self.n_particles, dim))
        pos[:, n_models] = np.random.uniform(0.50, 0.58, self.n_particles)
        
        # Velocity initialization
        vel = np.random.uniform(-0.05, 0.05, (self.n_particles, dim))
        
        pbest_pos = pos.copy()
        pbest_score = np.array([self._fitness(p, oof_probs, y_true) for p in pos])
        
        gbest_idx = np.argmax(pbest_score)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_score = pbest_score[gbest_idx]
        
        self.history_ = [gbest_score]
        
        for it in range(self.n_iter):
            r1 = np.random.rand(self.n_particles, dim)
            r2 = np.random.rand(self.n_particles, dim)
            
            # Velocity update
            vel = (
                self.w_inertia * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            # Position update
            pos += vel
            
            # Box constraints: weights in [0.05, 1.0], threshold in [0.48, 0.62]
            pos[:, :n_models] = np.clip(pos[:, :n_models], 0.05, 1.0)
            pos[:, n_models] = np.clip(pos[:, n_models], 0.48, 0.62)
            
            # Fitness evaluation
            scores = np.array([self._fitness(p, oof_probs, y_true) for p in pos])
            improved = scores > pbest_score
            pbest_pos[improved] = pos[improved].copy()
            pbest_score[improved] = scores[improved]
            
            if np.max(scores) > gbest_score:
                gbest_score = np.max(scores)
                gbest_pos = pos[np.argmax(scores)].copy()
                
            self.history_.append(gbest_score)
            
        self.best_weights_ = gbest_pos[:n_models] / np.sum(gbest_pos[:n_models])
        self.best_threshold_ = gbest_pos[n_models]
        self.best_fitness_ = gbest_score
        return self

    def predict_proba(self, test_probs):
        return np.dot(test_probs, self.best_weights_)

    def predict(self, test_probs):
        probs = self.predict_proba(test_probs)
        return (probs >= self.best_threshold_).astype(int)
