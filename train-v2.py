import pandas as pd
import numpy as np
import glob
import time
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import matplotlib.pyplot as plt


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BINS_CALIBRATION = 10
CV_FOLDS = 10
ENERGY_ERR_CUT = 1e-8
EPS = 1e-15
G = 4 * np.pi**2


def load_data(data_path):
    files = glob.glob(data_path)
    if not files:
        raise FileNotFoundError(f"No files found at {data_path}")
    dfs = [pd.read_csv(f) for f in tqdm(files, desc="Loading files")]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} files")
    return df

def filter_unphysical_binaries(df_bin, min_orbits=1, max_ecc=0.99999):    
    # 1. Require at least min_orbits complete orbits
    mask_orbits = df_bin['p'] < (df_bin['sim_time'] / min_orbits)    
    # 2. Filter extreme eccentricities (numerical artifacts)
    mask_ecc = df_bin['e'] < max_ecc    
    # Combine all filters
    mask = mask_orbits & mask_ecc    
    n_before = len(df_bin)
    n_after = mask.sum()    
    print(f"Filtered unphysical binaries:")
    print(f"  Before: {n_before:,}")
    print(f"  After:  {n_after:,} ({n_after/n_before*100:.1f}% retained)")  
    return df_bin[mask].copy()

def clean_data(df):
    initial = len(df)    
    # Filter by energy error
    if 'sim_e_err' in df.columns:
        df = df[df['sim_e_err'] < ENERGY_ERR_CUT]    
    # Separate binaries and non-binaries
    mask_no_binary = df['bin'] == False    
    # Filter binaries using the complete function
    df_bin = df[df['bin'] == True].copy()
    df_bin_filtered = filter_unphysical_binaries(df_bin, 1, 0.9998)    
    # Combine: keep all non-binaries + filtered binaries
    df = pd.concat([df[mask_no_binary], df_bin_filtered], ignore_index=True)    
    print(f"Cleaned: {initial:,} -> {len(df):,} rows ({len(df)/initial*100:.1f}% retained)")
    return df

def _norm(x):
    return np.linalg.norm(x, axis=1)

def _mass_entropy(m1, m2, m3, M):
    p1, p2, p3 = m1/M, m2/M, m3/M
    return abs(p1 * np.log(p1) + p2 * np.log(p2) + p3 * np.log(p3))

def _compute_pairwise_vectors(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """Compute relative velocity and position vectors and their magnitudes"""
    # Relative velocities
    vx12, vx13, vx23 = v1x - v2x, v1x - v3x, v2x - v3x
    vy12, vy13, vy23 = v1y - v2y, v1y - v3y, v2y - v3y
    vz12, vz13, vz23 = v1z - v2z, v1z - v3z, v2z - v3z    
    # Relative positions
    x12, x13, x23 = x1 - x2, x1 - x3, x2 - x3
    y12, y13, y23 = y1 - y2, y1 - y3, y2 - y3
    z12, z13, z23 = z1 - z2, z1 - z3, z2 - z3    
    # Vectors
    v12_vec = np.column_stack([vx12, vy12, vz12])
    v13_vec = np.column_stack([vx13, vy13, vz13])
    v23_vec = np.column_stack([vx23, vy23, vz23])    
    r12_vec = np.column_stack([x12, y12, z12])
    r13_vec = np.column_stack([x13, y13, z13])
    r23_vec = np.column_stack([x23, y23, z23])    
    # Magnitudes
    v12 = np.linalg.norm(v12_vec, axis=1)
    v13 = np.linalg.norm(v13_vec, axis=1)
    v23 = np.linalg.norm(v23_vec, axis=1)    
    r12 = np.linalg.norm(r12_vec, axis=1)
    r13 = np.linalg.norm(r13_vec, axis=1)
    r23 = np.linalg.norm(r23_vec, axis=1)    
    return v12, v13, v23, r12, r13, r23

def _compute_mass_features(m1, m2, m3, M_tot):
    """Compute mass-related features"""
    m_min = np.minimum(np.minimum(m1, m2), m3)
    m_max = np.maximum(np.maximum(m1, m2), m3)
    m_mid = M_tot - m_min - m_max    
    # Indices
    m_min_idx = np.argmin(np.column_stack([m1, m2, m3]), axis=1)
    m_max_idx = np.argmax(np.column_stack([m1, m2, m3]), axis=1)
    m_mid_idx = 3 - m_min_idx - m_max_idx    
    features = {
        'm_min': m_min, 'm_mid': m_mid, 'm_max': m_max,
        'm_min_idx': m_min_idx, 'm_mid_idx': m_mid_idx, 'm_max_idx': m_max_idx,
        'm_min_frac': m_min / (M_tot + EPS),
        'm_max_frac': m_max / (M_tot + EPS),
        'm_mid_frac': m_mid / (M_tot + EPS),
        'mass_entropy': _mass_entropy(m1, m2, m3, M_tot),
        'mass_pair_extreme': (m_min * m_max + EPS),
        'mass_pair_mid_min': (m_mid * m_min + EPS),
        'mass_ratio_mid_min_max': (m_mid + m_min) / (m_max + EPS),
        'mass_hierarchy': (m_max / (m_min + EPS) + EPS),
        'mass_asymmetry': (m_max - m_min) / (m_max + m_min + EPS)
    }
    return features

def _compute_velocity_features(m1, m2, m3, v1, v2, v3, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, r_enc, M_tot):
    """Compute velocity-related features"""
    v_mean = (v1 + v2 + v3) / 3
    v_esc = np.sqrt(2 * G * M_tot / (r_enc + EPS))    
    # Directional cosines
    norm1 = np.sqrt(v1x**2 + v1y**2 + v1z**2) + EPS
    norm2 = np.sqrt(v2x**2 + v2y**2 + v2z**2) + EPS
    norm3 = np.sqrt(v3x**2 + v3y**2 + v3z**2) + EPS    
    cos12 = (v1x*v2x + v1y*v2y + v1z*v2z) / (norm1 * norm2)
    cos13 = (v1x*v3x + v1y*v3y + v1z*v3z) / (norm1 * norm3)
    cos23 = (v2x*v3x + v2y*v3y + v2z*v3z) / (norm2 * norm3)
    cos_mean = (cos12 + cos13 + cos23) / 3    
    return {
        'v_mean': v_mean,
        'v_esc': v_esc,
        'v_esc_mean': (v_esc / (v_mean + EPS) + EPS),
        'cos_mean': cos_mean
    }

def _compute_impact_features(b1, b2, b3, r_enc):
    """Compute impact parameter features"""
    b_mean = (b1 + b2 + b3) / 3
    return {
        'b_mean': b_mean,
        'b_mean_r': b_mean / (r_enc + EPS),
        'b_hierarchy': np.maximum(np.maximum(b1, b2), b3) / (np.minimum(np.minimum(b1, b2), b3) + EPS)
    }

def _compute_focus_features(M_tot, b_mean, v_mean):
    """Compute gravitational focusing features"""
    focus = G * M_tot / (b_mean * v_mean**2 + EPS)
    return {
        'focus': (focus)
    }

def _compute_energy_features(m1, m2, m3, v1, v2, v3, r12, r13, r23):
    """Compute energy-related features using exact pairwise separations"""
    ke = 0.5 * (m1*v1**2 + m2*v2**2 + m3*v3**2)
    pe = -G * (m1*m2 / (r12 + EPS) + m1*m3 / (r13 + EPS) + m2*m3 / (r23 + EPS))
    hardness = np.abs(pe) / (ke + EPS)
    return ke, pe, hardness

def _compute_angular_features(m1, m2, m3, x1, y1, z1, x2, y2, z2, x3, y3, z3, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, ke, pe, M_tot):
    """Compute angular momentum features"""
    # Center of mass
    cm_x = (m1*x1 + m2*x2 + m3*x3) / (M_tot + EPS)
    cm_y = (m1*y1 + m2*y2 + m3*y3) / (M_tot + EPS)
    cm_z = (m1*z1 + m2*z2 + m3*z3) / (M_tot + EPS)    
    cm_vx = (m1*v1x + m2*v2x + m3*v3x) / (M_tot + EPS)
    cm_vy = (m1*v1y + m2*v2y + m3*v3y) / (M_tot + EPS)
    cm_vz = (m1*v1z + m2*v2z + m3*v3z) / (M_tot + EPS)    
    # Relative positions and velocities
    rx1, ry1, rz1 = x1 - cm_x, y1 - cm_y, z1 - cm_z
    rx2, ry2, rz2 = x2 - cm_x, y2 - cm_y, z2 - cm_z
    rx3, ry3, rz3 = x3 - cm_x, y3 - cm_y, z3 - cm_z    
    rvx1, rvy1, rvz1 = v1x - cm_vx, v1y - cm_vy, v1z - cm_vz
    rvx2, rvy2, rvz2 = v2x - cm_vx, v2y - cm_vy, v2z - cm_vz
    rvx3, rvy3, rvz3 = v3x - cm_vx, v3y - cm_vy, v3z - cm_vz    
    # Angular momentum components
    Lx = m1*(ry1*rvz1 - rz1*rvy1) + m2*(ry2*rvz2 - rz2*rvy2) + m3*(ry3*rvz3 - rz3*rvy3)
    Ly = m1*(rz1*rvx1 - rx1*rvz1) + m2*(rz2*rvx2 - rx2*rvz2) + m3*(rz3*rvx3 - rx3*rvz3)
    Lz = m1*(rx1*rvy1 - ry1*rvx1) + m2*(rx2*rvy2 - ry2*rvx2) + m3*(rx3*rvy3 - ry3*rvx3)
    L_tot = np.sqrt(Lx**2 + Ly**2 + Lz**2)    
    return (L_tot / abs(ke + pe) + EPS)

def _compute_virial_features(m1, m2, m3, ke, pe, r_enc, M_tot):
    """Compute virial radius cut-off features (Ginat & Perets 2024)"""
    M2_sq = (m1*m2 + m1*m3 + m2*m3) / 3
    RE = G * M2_sq / (2 * (ke + np.abs(pe)) + EPS)
    r_enc_over_RE = r_enc / (RE)
    return r_enc_over_RE

def _extract_by_mass_order(m1, m2, m3, v1, v2, v3, b1, b2, b3, theta1, theta2, theta3, phi1, phi2, phi3, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z):
    """Extract values ordered by min/mid/max mass"""
    M_tot = m1 + m2 + m3
    m_min_idx = np.argmin(np.column_stack([m1, m2, m3]), axis=1)
    m_max_idx = np.argmax(np.column_stack([m1, m2, m3]), axis=1)
    m_mid_idx = 3 - m_min_idx - m_max_idx    
    n = len(m_min_idx)
    idx = np.arange(n)    
    velocities = np.column_stack([v1, v2, v3])
    b_mags = np.column_stack([b1, b2, b3])
    thetas = np.column_stack([theta1, theta2, theta3])
    phis = np.column_stack([phi1, phi2, phi3])    
    # Velocity vectors for each star
    vx_all = np.column_stack([v1x, v2x, v3x])
    vy_all = np.column_stack([v1y, v2y, v3y])
    vz_all = np.column_stack([v1z, v2z, v3z])    
    # Extract vectors for min, mid, max stars
    v_min_x = vx_all[idx, m_min_idx]
    v_min_y = vy_all[idx, m_min_idx]
    v_min_z = vz_all[idx, m_min_idx]    
    v_mid_x = vx_all[idx, m_mid_idx]
    v_mid_y = vy_all[idx, m_mid_idx]
    v_mid_z = vz_all[idx, m_mid_idx]    
    v_max_x = vx_all[idx, m_max_idx]
    v_max_y = vy_all[idx, m_max_idx]
    v_max_z = vz_all[idx, m_max_idx]    
    # Stack into vectors
    v_min_vec = np.column_stack([v_min_x, v_min_y, v_min_z])
    v_mid_vec = np.column_stack([v_mid_x, v_mid_y, v_mid_z])
    v_max_vec = np.column_stack([v_max_x, v_max_y, v_max_z])    
    # Magnitudes
    v_min_mag = np.linalg.norm(v_min_vec, axis=1)
    v_mid_mag = np.linalg.norm(v_mid_vec, axis=1)
    v_max_mag = np.linalg.norm(v_max_vec, axis=1)    
    # Pairwise angles
    dot_min_mid = np.sum(v_min_vec * v_mid_vec, axis=1)
    dot_min_max = np.sum(v_min_vec * v_max_vec, axis=1)
    dot_mid_max = np.sum(v_mid_vec * v_max_vec, axis=1)    
    angle_min_mid = np.arccos(np.clip(dot_min_mid / (v_min_mag * v_mid_mag + EPS), -1, 1))
    angle_min_max = np.arccos(np.clip(dot_min_max / (v_min_mag * v_max_mag + EPS), -1, 1))
    angle_mid_max = np.arccos(np.clip(dot_mid_max / (v_mid_mag * v_max_mag + EPS), -1, 1))    
    return {
        'v_min': velocities[idx, m_min_idx],
        'v_mid': velocities[idx, m_mid_idx],
        'v_max': velocities[idx, m_max_idx],
        'b_min': b_mags[idx, m_min_idx],
        'b_mid': b_mags[idx, m_mid_idx],
        'b_max': b_mags[idx, m_max_idx],
        'theta_min': thetas[idx, m_min_idx],
        'theta_mid': thetas[idx, m_mid_idx],
        'theta_max': thetas[idx, m_max_idx],
        'phi_min': phis[idx, m_min_idx],
        'phi_mid': phis[idx, m_mid_idx],
        'phi_max': phis[idx, m_max_idx],
        'angle_min_mid': angle_min_mid,
        'angle_min_max': angle_min_max,
        'angle_mid_max': angle_mid_max,
        'angle_mean': (angle_min_mid + angle_min_max + angle_mid_max) / 3
    }

def build_features(df):
    # ========== READ BASIC DATA ==========
    m1, m2, m3 = df['m1'].values, df['m2'].values, df['m3'].values
    r_enc = df['r_enc'].values    
    # Impact parameters
    b1 = np.sqrt(df['bx1']**2 + df['by1']**2 + df['bz1']**2)
    b2 = np.sqrt(df['bx2']**2 + df['by2']**2 + df['bz2']**2)
    b3 = np.sqrt(df['bx3']**2 + df['by3']**2 + df['bz3']**2)    
    # Velocities
    v1 = np.sqrt(df['vxi1']**2 + df['vyi1']**2 + df['vzi1']**2)
    v2 = np.sqrt(df['vxi2']**2 + df['vyi2']**2 + df['vzi2']**2)
    v3 = np.sqrt(df['vxi3']**2 + df['vyi3']**2 + df['vzi3']**2)    
    # Components for vectors
    v1x, v1y, v1z = df['vxi1'], df['vyi1'], df['vzi1']
    v2x, v2y, v2z = df['vxi2'], df['vyi2'], df['vzi2']
    v3x, v3y, v3z = df['vxi3'], df['vyi3'], df['vzi3']    
    # Positions
    x1, y1, z1 = df['xi1'], df['yi1'], df['zi1']
    x2, y2, z2 = df['xi2'], df['yi2'], df['zi2']
    x3, y3, z3 = df['xi3'], df['yi3'], df['zi3']    
    # Angles
    theta1, theta2, theta3 = df['theta1'], df['theta2'], df['theta3']
    phi1, phi2, phi3 = df['phi1'], df['phi2'], df['phi3']    
    M_tot = m1 + m2 + m3    
    # ========== COMPUTE FEATURES ==========    
    # Mass features
    mass = _compute_mass_features(m1, m2, m3, M_tot)    
    # Pairwise vectors
    v12, v13, v23, r12, r13, r23 = _compute_pairwise_vectors(
        v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z,
        x1, y1, z1, x2, y2, z2, x3, y3, z3
    )    
    # Impact features
    impact = _compute_impact_features(b1, b2, b3, r_enc)    
    # Velocity features
    vel = _compute_velocity_features(m1, m2, m3, v1, v2, v3, 
                                       v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z,
                                       r_enc, M_tot)    
    # Energy features
    ke, pe, hardness = _compute_energy_features(m1, m2, m3, v1, v2, v3, r12, r13, r23)    
    # Focus features
    focus = _compute_focus_features(M_tot, impact['b_mean'], vel['v_mean'])    
    # Angular features
    L_ke_ratio = _compute_angular_features(
        m1, m2, m3, x1, y1, z1, x2, y2, z2, x3, y3, z3,
        v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, ke, pe, M_tot
    )    
    # Virial features
    r_enc_over_RE = _compute_virial_features(m1, m2, m3, ke, pe, r_enc, M_tot)    
    # Mass-ordered values
    ordered = _extract_by_mass_order(m1, m2, m3, v1, v2, v3, b1, b2, b3,
                                        theta1, theta2, theta3, phi1, phi2, phi3,
                                            v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z)
    # ========== ASSEMBLE DATAFRAME ==========
    X = pd.DataFrame({
        'm_min': np.log10(mass['m_min']),
        'm_mid': np.log10(mass['m_mid']),
        'm_max': np.log10(mass['m_max']),
        'M_tot': np.log10(mass['m_min'] + mass['m_mid'] + mass['m_max']),
        'm_min_frac': np.log10(mass['m_min_frac']),
        'm_mid_frac': np.log10(mass['m_mid_frac']),
        'm_max_frac': np.log10(mass['m_max_frac']),
        'mass_entropy': np.log10(mass['mass_entropy']),
        'mass_ratio_mid_min_max': np.log10(mass['mass_ratio_mid_min_max']),
        'mass_hierarchy': np.log10(mass['mass_hierarchy']),
        'mass_asymmetry': np.log10(mass['mass_asymmetry']),
        'v_esc_mean': np.log10(vel['v_esc_mean']),
        'focus': np.log10(focus['focus']),
        'hardness': np.log10(hardness),
        'L_ke_ratio': np.log10(L_ke_ratio),
        'r_enc_over_RE': np.log10(r_enc_over_RE),
        'v_min': np.log10(ordered['v_min']),
        'v_mid': np.log10(ordered['v_mid']),
        'v_max': np.log10(ordered['v_max']),
        'v_min_frac': np.log10(ordered['v_min']/vel['v_esc']),
        'v_mid_frac': np.log10(ordered['v_mid']/vel['v_esc']),
        'v_max_frac': np.log10(ordered['v_max']/vel['v_esc']),
        'v_hierarchy': np.log10(ordered['v_max']/ordered['v_min']),
        'v_asymmetry': np.log10(abs(ordered['v_max'] - ordered['v_min'])/(ordered['v_max'] + ordered['v_min'])),
        'b_min': np.log10(ordered['b_min']),
        'b_mid': np.log10(ordered['b_mid']),
        'b_max': np.log10(ordered['b_max']),
        'cos_angle_min_mid': np.cos(ordered['angle_min_mid']),
        'cos_angle_min_max': np.cos(ordered['angle_min_max']),
        'cos_angle_mid_max': np.cos(ordered['angle_mid_max']),
        'cos_mean_angles': vel['cos_mean']
    })    
    return X

def balance_dataset(X, y):
    df_temp = X.copy()
    df_temp['bin'] = y    
    df_majority = df_temp[df_temp.bin == False]
    df_minority = df_temp[df_temp.bin == True]    
    if len(df_minority) == 0:
        return X, y    
    df_majority_down = df_majority.sample(n=len(df_minority), random_state=RANDOM_STATE)
    df_balanced = pd.concat([df_majority_down, df_minority]).sample(frac=1, random_state=RANDOM_STATE)    
    return df_balanced.drop(columns=['bin']), df_balanced['bin'].values

def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': tn / (tn + fp + EPS),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'brier': brier_score_loss(y_true, y_proba),
        'ece': compute_ece(y_true, y_proba)
    }

def compute_ece(y_true, y_proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    ece = 0.0
    for i in range(len(prob_true)):
        if i == 0:
            mask = y_proba <= prob_pred[i]
        else:
            mask = (y_proba > prob_pred[i-1]) & (y_proba <= prob_pred[i])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_true)) * abs(prob_pred[i] - y_true[mask].mean())
    return ece

def evaluate_by_r_enc(df, X_test, y_test, model, test_indices):
    r_enc_bins = [(0.01, 1), (1, 100), (100, 10000)]
    bin_names = ["(0.01, 1)", "(1, 100)", "(100, 10000)"]    
    print("\n" + "="*80)
    print("PERFORMANCE BY ENCOUNTER RADIUS")
    print("="*80)
    print(f"{'r_enc':<15} {'n':<10} {'Binaries':<12} {'Frac':<8} {'Acc':<8} {'PR-AUC':<10}")
    print("-"*80)    
    results = []
    for (low, high), name in zip(r_enc_bins, bin_names):
        mask = (df['r_enc'].values >= low) & (df['r_enc'].values < high)
        mask_test = mask[test_indices]        
        n = mask_test.sum()
        if n < 50:
            continue        
        y_sub = y_test[mask_test]
        X_sub = X_test[mask_test]
        bin_frac = y_sub.mean()
        acc = accuracy_score(y_sub, model.predict(X_sub))
        pr_auc = average_precision_score(y_sub, model.predict_proba(X_sub)[:,1])        
        print(f"{name:<15} {n:<10,} {int(bin_frac*n):<12,} {bin_frac:<8.3f} {acc:<8.4f} {pr_auc:<10.4f}")
        results.append({'r_enc_bin': name, 'n': n, 'binary_fraction': bin_frac, 'accuracy': acc, 'pr_auc': pr_auc})    
    print("="*80)
    return results

def evaluate_by_hardness(df, X_test, y_test, model, test_indices):    
    # Compute exact hardness
    m1, m2, m3 = df['m1'].values, df['m2'].values, df['m3'].values
    v1_ini = np.sqrt(df['vxi1'].values**2 + df['vyi1'].values**2 + df['vzi1'].values**2)
    v2_ini = np.sqrt(df['vxi2'].values**2 + df['vyi2'].values**2 + df['vzi2'].values**2)
    v3_ini = np.sqrt(df['vxi3'].values**2 + df['vyi3'].values**2 + df['vzi3'].values**2)    
    x1, y1, z1 = df['xi1'].values, df['yi1'].values, df['zi1'].values
    x2, y2, z2 = df['xi2'].values, df['yi2'].values, df['zi2'].values
    x3, y3, z3 = df['xi3'].values, df['yi3'].values, df['zi3'].values    
    r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    r13 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
    r23 = np.sqrt((x2-x3)**2 + (y2-y3)**2 + (z2-z3)**2)    
    KE = 0.5 * (m1*v1_ini**2 + m2*v2_ini**2 + m3*v3_ini**2)
    PE = -G * (m1*m2/(r12+EPS) + m1*m3/(r13+EPS) + m2*m3/(r23+EPS))
    hardness = np.abs(PE) / (KE + EPS)
    bins = [(0.0,0.01), (0.01,0.1), (0.1,1.0)]
    names = ["0.0-0.01", "0.01-0.1", "0.1-1.0"]    
    print("\n" + "="*80)
    print("PERFORMANCE BY HARDNESS")
    print("="*80)
    print(f"{'Hardness':<12} {'n':<10} {'Binaries':<12} {'Frac':<8} {'Acc':<8} {'PR-AUC':<10}")
    print("-"*80)    
    results = []
    for (low, high), name in zip(bins, names):
        mask = (hardness >= low) & (hardness < high)
        mask_test = mask[test_indices]        
        n = mask_test.sum()
        if n < 50:
            continue        
        y_sub = y_test[mask_test]
        X_sub = X_test[mask_test]
        bin_frac = y_sub.mean()
        acc = accuracy_score(y_sub, model.predict(X_sub))
        pr_auc = average_precision_score(y_sub, model.predict_proba(X_sub)[:,1])        
        print(f"{name:<12} {n:<10,} {int(bin_frac*n):<12,} {bin_frac:<8.3f} {acc:<8.4f} {pr_auc:<10.4f}")
        results.append({'hardness_bin': name, 'n': n, 'binary_fraction': bin_frac, 'accuracy': acc, 'pr_auc': pr_auc})    
    print("="*80)
    return results

def physical_baseline_test(X, y, save_path='baseline_results.csv'):    
    # Define physical directions for top 10 features
    physical_directions = {
        'mass_hierarchy': '<',      # Extreme mass ratios → fewer binaries
        'mass_asymmetry': '<',      # Asymmetric masses → fewer binaries
        'hardness': '>',            # Higher hardness → more binaries
        'v_esc_mean': '>',          # Higher escape velocity → more binaries
        'v_min_frac': '<',          # Slow lightest star → more binaries
        'mass_entropy': '<',        # Low entropy (unequal masses) → fewer binaries
        'r_enc_over_RE': '<',       # Smaller r_enc → stronger interaction → more binaries
        'v_hierarchy': '<',         # Extreme velocity hierarchy → fewer binaries
        'v_asymmetry': '<',         # Asymmetric velocities → fewer binaries
        'v_mid_frac': '<',          # Slow intermediate star → more binaries
    }    
    results = {}    
    print("\n" + "="*80)
    print("PHYSICAL BASELINE COMPARISON (Physically Motivated Directions)")
    print("="*80)
    print(f"{'Feature':<25} {'Direction':<12} {'Threshold':<15} {'PR-AUC':<12} {'Accuracy':<12}")
    print("-"*80)    
    for feature, direction in physical_directions.items():
        # Check if feature exists
        if feature not in X.columns:
            print(f"⚠️  Warning: '{feature}' not found in X. Skipping.")
            continue        
        # Median threshold
        threshold = X[feature].median()        
        # Predict based on physical direction
        if direction == '>':
            y_pred = (X[feature] > threshold).astype(int)
        else:  # direction == '<'
            y_pred = (X[feature] < threshold).astype(int)        
        # Compute metrics
        pr_auc = average_precision_score(y, y_pred)
        acc = accuracy_score(y, y_pred)        
        results[feature] = {
            'direction': direction,
            'threshold': threshold,
            'pr_auc': pr_auc,
            'accuracy': acc
        }        
        print(f"{feature:<25} {direction:<12} {threshold:<15.6f} {pr_auc:<12.4f} {acc:<12.4f}")    
    # Save to CSV
    df_results = pd.DataFrame([
        {
            'feature': feat,
            'direction': res['direction'],
            'threshold': res['threshold'],
            'pr_auc': res['pr_auc'],
            'accuracy': res['accuracy']
        }
        for feat, res in results.items()
    ])
    df_results = df_results.sort_values('pr_auc', ascending=False)
    df_results.to_csv(save_path, index=False)
    print(f"\n✅ Results saved to {save_path}")    
    return results

def save_results(model, X, y_test, y_pred, y_proba, metrics, cv_scores, 
                 r_enc_results, hardness_results, baseline_results, save_dir='./results/'):    
    os.makedirs(save_dir, exist_ok=True)    
    # 1. Feature importance
    feat_imp = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    pd.DataFrame(feat_imp, columns=['Feature', 'Importance']).to_csv(f'{save_dir}feature_importance.csv', index=False)    
    # --- NEW: Save the trained model ---
    # Save in JSON format (recommended for XGBoost)
    model.save_model(f'{save_dir}xgboost_model.json')
    print(f"✓ Model saved to {save_dir}xgboost_model.json")    
    # Optional: Save as binary (UBJ) format for faster loading
    model.save_model(f'{save_dir}xgboost_model.ubj')    
    # 2. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['True No Binary', 'True Binary'], columns=['Pred No Binary', 'Pred Binary'])
    cm_df.to_csv(f'{save_dir}confusion_matrix.csv', index=False)    
    # 3. Calibration data
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=N_BINS_CALIBRATION, strategy='uniform')
    cal_df = pd.DataFrame({'predicted_probability': prob_pred, 'fraction_of_positives': prob_true})
    cal_df.to_csv(f'{save_dir}calibration_data.csv', index=False)    
    # 4. Prediction probabilities (confidence)
    prob_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'probability_binary': y_proba
    })
    prob_df.to_csv(f'{save_dir}prediction_probabilities.csv', index=False)    
    # 5. Test metrics
    pd.DataFrame([metrics]).to_csv(f'{save_dir}test_metrics.csv', index=False)    
    # 6. CV results
    pd.DataFrame([{'cv_pr_auc_mean': cv_scores.mean(), 'cv_pr_auc_std': cv_scores.std()}]).to_csv(f'{save_dir}cv_results.csv', index=False)    
    # 7. r_enc results
    if r_enc_results:
        pd.DataFrame(r_enc_results).to_csv(f'{save_dir}r_enc_performance.csv', index=False)    
    # 8. Hardness results
    if hardness_results:
        pd.DataFrame(hardness_results).to_csv(f'{save_dir}hardness_performance.csv', index=False)        
    print(f"\n✅ All results saved to {save_dir}")

def main():
    DATA_PATH = "./data/split/dataset_*.csv"  # Adjust to your data path    
    print("\n" + "="*60)
    print("BINARY FORMATION PREDICTION - XGBoost Classifier")
    print("="*60)    
    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_data(DATA_PATH)    
    # 2. Clean
    print("\n[2/5] Cleaning data...")
    df = clean_data(df)
    # 3. Build features
    print("\n[3/5] Building features...")
    X = build_features(df)
    y = df['bin'].values
    print(f"Features: {X.shape[1]}, Binary rate: {y.mean():.4f} ({y.sum():,} positives)")    
    # 4. Balance
    print("\n[4/5] Balancing data...")
    X, y = balance_dataset(X, y)
    print(f"Balanced: {len(X):,} samples (50/50)")    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    baseline_results = physical_baseline_test(X_test, y_test, save_path='./results/baseline_comparison.csv')
    # 5. Train XGBoost
    print("\n[5/5] Training XGBoost...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + EPS)    
    model = xgb.XGBClassifier(
        n_estimators=1200, max_depth=18, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.9, objective='binary:logistic',
        scale_pos_weight=scale_pos_weight, reg_lambda=1.0,
        tree_method='hist', device='cuda', eval_metric='logloss',
        n_jobs=-1, random_state=RANDOM_STATE
    )    
    start = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"Training time: {time.time()-start:.2f}s")    
    # Predict (probabilities)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]    
    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"ECE:       {metrics['ece']:.4f}")    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}   FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}   TP: {cm[1,1]:,}")    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='average_precision')
    print(f"\nCV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")    
    # Feature importance
    feat_imp = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    for name, imp in feat_imp[:17]:
        print(f"  {name:<25} {imp:.4f}")    
    test_indices = X_test.index    
    r_enc_results = evaluate_by_r_enc(df, X_test, y_test, model, test_indices)
    hardness_results = evaluate_by_hardness(df, X_test, y_test, model, test_indices)    
    # Speed benchmark (model only)
    X_sample = X.sample(min(100000, len(X)), random_state=RANDOM_STATE)
    start = time.perf_counter()
    model.predict(X_sample)
    speed = len(X_sample) / (time.perf_counter() - start)
    print(f"\nSpeed: {speed:,.0f} predictions/sec")    
    # Save all results
    save_results(model, X, y_test, y_pred, y_proba, metrics, cv_scores, 
                 r_enc_results, hardness_results, baseline_results, save_dir='./results/')    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()