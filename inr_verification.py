import numpy as np
import matplotlib.pyplot as plt

# ======================
# Constants / Parameters
# ======================
EkT = -174    # dBm/Hz
B = 100e6
Tx_power_dbm = 30 # dBm
Tx_power = 10 ** ((Tx_power_dbm - 30) / 10)  # W

NF_vsat = 2   # dB
NF_bs = 3     # dB
preamble_time = 20e-6 

# Total receiver noise powers across B
N0_vsat =  10 ** ((EkT + 10 * np.log10(B) + NF_vsat - 30) / 10)
N0_bs   =  10 ** ((EkT + 10 * np.log10(B) + NF_bs   - 30) / 10)

# Channel-estimation noise density mapped to per-antenna estimate variance:
# h_hat = h + CN(0, N0_sigma * I), with N0_sigma = N0_bs / (Tx_power * preamble_time)
N0_sigma =  10 ** ((EkT + NF_bs - 30) / 10) / Tx_power / preamble_time

# ==============
# Array geometry
# ==============
dsep = 0.5  # lambda
theta0_rad = 0.0
theta1_rad = np.deg2rad(15.0)

ntx_num = np.array([16, 32, 48, 64])

# =====================
# Monte Carlo / Sweeps
# =====================
simu_num = 200
gamma_num = 120
gamma_h1 = np.logspace(-8, -3, gamma_num)

# Storage
inr_mc_mean = np.zeros((len(ntx_num), gamma_num), dtype=float)
inr_theory  = np.zeros((len(ntx_num), gamma_num), dtype=float)
inr_part_D  = np.zeros((len(ntx_num), gamma_num), dtype=float)
inr_part_V  = np.zeros((len(ntx_num), gamma_num), dtype=float)

def beamforming_vec(h0, h1_hat):
    """Construct w per closed-form using h0 and h1_hat."""
    C0 = np.linalg.norm(h0)**2
    C1 = np.linalg.norm(h1_hat)**2
    rho = (h1_hat.conj().T @ h0) / np.sqrt(C1 * C0)
    alpha0 = 1.0 / (np.sqrt(1 - np.abs(rho)**2) * np.sqrt(C0))
    alpha1 = -rho / (np.sqrt(1 - np.abs(rho)**2) * np.sqrt(C1))
    w = alpha0 * h0 + alpha1 * h1_hat
    return w, rho.item()

rng = np.random.default_rng(12345)

for idx_ntx, ntx in enumerate(ntx_num):
    # steering vectors (unit-norm across antennas due to 1/sqrt(N_t))
    phase0 = 2 * np.pi * np.arange(ntx) * dsep * np.cos(theta0_rad)
    phase1 = 2 * np.pi * np.arange(ntx) * dsep * np.cos(theta1_rad)
    h0 = np.exp(1j * phase0).reshape(-1,1) / np.sqrt(ntx)
    h1_base = np.exp(1j * phase1).reshape(-1,1) / np.sqrt(ntx)

    # correlation between directions (independent of gamma scale)
    rho_true = (h1_base.conj().T @ h0) / (np.linalg.norm(h1_base) * np.linalg.norm(h0))
    rho_true = rho_true.item()
    abs_rho2 = np.abs(rho_true)**2

    G0 = (np.linalg.norm(h0)**2) / ntx  # = 1/N_t with current normalization

    for g, gamma in enumerate(gamma_h1):
        h1 = gamma * h1_base

        # Monte Carlo
        vals = np.empty(simu_num, dtype=float)
        for n in range(simu_num):
            noise = np.sqrt(N0_sigma/2.0) * (rng.standard_normal(h1.shape) + 1j * rng.standard_normal(h1.shape))
            h1_hat = h1 + noise
            w, _ = beamforming_vec(h0, h1_hat)
            vals[n] = (np.abs((w.conj().T @ h1).item())**2) * Tx_power / N0_vsat
        inr_mc_mean[idx_ntx, g] = vals.mean()

        # Theory using S, |D|^2, E|V|^2
        G1 = (np.linalg.norm(h1)**2) / ntx
        gamma_tr = Tx_power * preamble_time * G1 / N0_bs

        S = (1.0 / (ntx * G0)) * (1.0 + gamma_tr) / (1.0 + gamma_tr - gamma_tr * abs_rho2)
        Dsq = (ntx**2) * abs_rho2 * G0 * G1 / ((1.0 + gamma_tr)**2)
        EVsq = N0_sigma * (gamma_tr / (1.0 + gamma_tr))**2 * ntx * G0 * (1.0 + abs_rho2)

        inr_part_D[idx_ntx, g] = (Tx_power / N0_vsat) * S * Dsq
        inr_part_V[idx_ntx, g] = (Tx_power / N0_vsat) * S * EVsq
        inr_theory[idx_ntx, g]  = inr_part_D[idx_ntx, g] + inr_part_V[idx_ntx, g]

# ==========
# Plotting
# ==========
for idx, ntx in enumerate(ntx_num):
    plt.figure()
    plt.loglog(gamma_h1, inr_mc_mean[idx, :], label="MC mean")
    plt.loglog(gamma_h1, inr_theory[idx, :], linestyle="--", label="Theory")
    plt.xlabel("gamma (scale on h1)")
    plt.ylabel("INR")
    plt.title(f"INR vs gamma, N_t = {ntx}")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

# Split the theory components for largest array
idx = -1
plt.figure()
plt.loglog(gamma_h1, inr_theory[idx, :], label="Theory total")
plt.loglog(gamma_h1, inr_part_D[idx, :], linestyle="--", label="Term from |D|^2")
plt.loglog(gamma_h1, inr_part_V[idx, :], linestyle="-.", label="Term from E|V|^2")
plt.xlabel("gamma (scale on h1)")
plt.ylabel("INR (theory components)")
plt.title(f"Theory components vs gamma, N_t = {ntx_num[idx]}")
plt.legend()
plt.grid(True, which="both")
plt.show()

# Print a small error snapshot
sample_idxs = np.linspace(0, gamma_num-1, 6, dtype=int)
for idx, ntx in enumerate(ntx_num):
    mc_s = inr_mc_mean[idx, sample_idxs]
    th_s = inr_theory[idx, sample_idxs]
    rel_err = np.abs(mc_s - th_s) / np.maximum(mc_s, 1e-20)
    print(f"N_t={ntx} | gamma samples: {gamma_h1[sample_idxs]}")
    print("  MC mean:", mc_s)
    print("  Theory :", th_s)
    print("  RelErr :", rel_err, "\\n")
