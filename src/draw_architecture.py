import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram(output_path):
    """
    Generates a high-resolution publication-quality architecture diagram
    illustrating the 4 distinct tiers of HG-FASNet.
    """
    fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")
    
    # Color palette
    c_data = "#E0F2FE"      # Light Blue
    c_dpfr = "#E0E7FF"      # Indigo Tint
    c_attn = "#FEF3C7"      # Amber Tint
    c_base = "#DCFCE7"      # Emerald Tint
    c_pso  = "#FCE7F3"      # Rose Tint
    c_out  = "#F3E8FF"      # Purple Tint
    c_border = "#1E293B"    # Slate Border
    
    # Title
    ax.text(7.5, 8.55, "HG-FASNet: Hybrid Gated Feature-Attention Stacking Network", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#0F172A")
    ax.text(7.5, 8.2, "Novel Multi-Tier Clinical Diagnosis Architecture with Metaheuristic Swarm Optimization", 
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#475569")
    
    # Box helper
    def draw_box(x, y, w, h, bg_color, title, subtitle="", items=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                      facecolor=bg_color, edgecolor=c_border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#0F172A")
        if subtitle:
            ax.text(x + w/2, y + h - 0.7, subtitle, ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#334155")
        if items:
            for idx, item in enumerate(items):
                ax.text(x + 0.25, y + h - 1.1 - (idx * 0.35), f"• {item}", ha="left", va="center", fontsize=8.5, color="#1E293B")
                
    # Arrow helper
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#0284C7", lw=2.2, mutation_scale=15))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.2, label, ha="center", va="center", fontsize=8, fontweight="bold", color="#0369A1")

    # 1. Input Data Tier
    draw_box(0.5, 4.2, 2.5, 3.4, c_data, "Input Clinical Data", "Cleveland Heart Disease (13 Vars)",
             ["Demographics (Age, Sex)", "Chest Pain (CP Type 1-4)", "Vitals (trestbps, chol, fbs)", "Cardiac (ECG, thalach, exang)", "Imaging/Lab (oldpeak, slope, ca, thal)"])
             
    # 2. Tier 1: DPFR
    draw_box(3.5, 4.0, 3.2, 3.8, c_dpfr, "Tier 1: DPFR Module", "Dual-Perspective Refinement",
             ["Rate-Pressure Product: RPP = (BP*HR)/1000", "Cholesterol/Age Ratio: CAR", "Cardiac Pathological Burden (CPB)", "Exercise Ischemia Burden (EIB)", "Chest Pain Rate Index (CPM)", "RobustScaler (IQR-based outlier protection)"])
             
    # 3. Tier 2: MHC-FAG
    draw_box(7.2, 4.0, 3.2, 3.8, c_attn, "Tier 2: MHC-FAG Module", "Multi-Head Attention Gate",
             ["Scaled Dot-Product Self-Attention:", "  A = Softmax( Q^T K / sqrt(d) )", "Inter-Feature Interaction Context:", "  X_attn = X @ A", "Adaptive Gating Vector:", "  gamma = Sigmoid( X W_g + b_g )", "Gated Residual: gamma*X + (1-gamma)*X_attn"])

    # 4. Tier 3: Heterogeneous Base Learners
    draw_box(11.0, 4.6, 3.5, 3.2, c_base, "Tier 3: TA-HBL Learners", "Tri-Architectural Inductive Diversity",
             ["Model 1: LightGBM (Leaf-wise trees)", "Model 2: Random Forest (Bagged forests)", "Model 3: Calibrated RBF-SVM (Max-margin)"])
             
    # 5. Tier 4: PSO Metaheuristic Engine
    draw_box(11.0, 0.6, 3.5, 3.4, c_pso, "Tier 4: PSO Engine", "Swarm Calibration & Thresholding",
             ["5-Fold Out-of-Fold Validation", "Swarm Position: [w_1, w_2, w_3, tau*]", "Velocity Update: Inertia + Cognitive + Social", "Fitness: 0.35*F1 + 0.30*Acc + 0.20*Prec + 0.15*Rec", "Optimal w*: [0.41, 0.22, 0.36], tau*=0.58"])

    # 6. Final Diagnostic Output
    draw_box(5.0, 0.8, 4.5, 2.6, c_out, "Calibrated Diagnostic Decision", "Clinical Output & Stratification",
             ["Final Probability: P(Y=1) = Sum(w_m * p_m)", "Decision: Disease (1) if P >= tau* else Healthy (0)", "Verified Performance: 91.80% Acc | 92.59% Prec | 89.29% Rec", "Superior to Base Paper HRFLM (88.7% Acc, 90.1% Prec, 87.5% Rec)"])

    # Connectors
    draw_arrow(3.0, 5.9, 3.5, 5.9, "13 Raw Vars")
    draw_arrow(6.7, 5.9, 7.2, 5.9, "18 Eng. Vars")
    draw_arrow(10.4, 5.9, 11.0, 5.9, "Attended Matrix")
    draw_arrow(12.75, 4.6, 12.75, 4.0, "OOF Probs [p1,p2,p3]")
    draw_arrow(11.0, 2.1, 9.5, 2.1, "Optimized w*, tau*")
    
    # Backward feedback arrow from PSO to base model combination
    ax.annotate("", xy=(9.5, 1.4), xytext=(11.0, 1.4),
                arrowprops=dict(arrowstyle="->", color="#DB2777", lw=1.8, linestyle="--"))
    ax.text(10.25, 1.15, "Swarm Optimization", ha="center", va="center", fontsize=7.5, color="#9D174D", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[DrawArchitecture] Generated architecture diagram at {output_path}")

if __name__ == "__main__":
    generate_architecture_diagram("/home/harsha/Projects/results/architecture_workflow.png")
