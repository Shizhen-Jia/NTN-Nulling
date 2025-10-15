
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters (same as in the notebook run)
EkT = -174
B = 100e6
Tx_power_dbm = 30
Tx_power = 10 ** ((Tx_power_dbm - 30) / 10)

NF_vsat = 2
NF_bs = 3
preamble_time = 20e-6

N0_vsat =  10 ** ((EkT + 10 * np.log10(B) + NF_vsat - 30) / 10)
N0_bs   =  10 ** ((EkT + 10 * np.log10(B) + NF_bs   - 30) / 10)
N0_sigma =  10 ** ((EkT + NF_bs - 30) / 10) / Tx_power / preamble_time

dsep = 0.5
theta0 = 0.0
theta1 = 15.0
theta0_rad = np.deg2rad(theta0)
theta1_rad = np.deg2rad(theta1)

N_t = 32
trials = 1500
gammas = np.logspace(-7, -3, 7)

rng = np.random.default_rng(2025)

def steering_vec(N_t, theta_rad, dsep=0.5):
    phase = 2 * np.pi * np.arange(N_t) * dsep * np.cos(theta_rad)
    return np.exp(1j * phase).reshape(-1,1) / np.sqrt(N_t)

def beamforming_vec(h0, h1_hat):
    C0 = np.linalg.norm(h0)**2
    C1 = np.linalg.norm(h1_hat)**2
    rho = (h1_hat.conj().T @ h0) / np.sqrt(C1 * C0)
    alpha0 = 1.0 / (np.sqrt(1 - np.abs(rho)**2) * np.sqrt(C0))
    alpha1 = -rho / (np.sqrt(1 - np.abs(rho)**2) * np.sqrt(C1))
    w = alpha0 * h0 + alpha1 * h1_hat
    return w, rho.item()

h0 = steering_vec(N_t, theta0_rad, dsep)
h1_dir = steering_vec(N_t, theta1_rad, dsep)
G0 = (np.linalg.norm(h0)**2)/N_t
rho_true = (h1_dir.conj().T @ h0) / (np.linalg.norm(h1_dir) * np.linalg.norm(h0))
rho_true = rho_true.item()
abs_rho2 = np.abs(rho_true)**2

rows = []

for gamma in gammas:
    h1 = gamma * h1_dir
    G1 = (np.linalg.norm(h1)**2)/N_t
    gamma_r = Tx_power * preamble_time * G1 / N0_bs

    noise = (np.sqrt(N0_sigma/2.0) *
             (rng.standard_normal((N_t, trials)) + 1j * rng.standard_normal((N_t, trials))))
    h1_hat_all = h1 + noise

    norms2 = np.sum(np.abs(h1_hat_all)**2, axis=0)
    emp_hnorm = norms2.mean()/N_t
    th_hnorm  = G1 + N0_sigma

    z10 = (noise.conj().T @ h0).reshape(-1) / N_t
    emp_B10 = np.mean(np.abs(z10)**2)
    th_B10  = (G0 * N0_sigma) / N_t

    num = (h1_hat_all.conj().T @ h0).reshape(-1)
    den = (np.linalg.norm(h0) * np.linalg.norm(h1_hat_all, axis=0))
    rho_hat_all = num / den
    emp_E_rho2 = np.mean(np.abs(rho_hat_all)**2)
    alpha = gamma_r / (1.0 + gamma_r)
    th_E_rho2 = alpha * (abs_rho2 + 1.0/(N_t * gamma_r))

    S_def = 1.0 / (np.linalg.norm(h0)**2 * (1.0 - emp_E_rho2))
    S_approx = (1.0/(N_t*G0)) * (1.0 + gamma_r) / (1.0 + gamma_r - gamma_r*abs_rho2)

    lhs = np.mean([ (h0.conj().T @ noise[:,k:k+1] @ noise[:,k:k+1].conj().T @ h1).item()
                    for k in range(trials) ])
    rhs = (N0_sigma * (h0.conj().T @ h1).item())

    E_hhat_norm2 = norms2.mean()
    a = (h0.conj().T @ noise) * (np.linalg.norm(h1)**2)
    b = (h1.conj().T @ h0).item() * (h1.conj().T @ noise).reshape(-1)
    V_all = (a + np.conj(b)) / E_hhat_norm2
    emp_E_V2 = np.mean(np.abs(V_all)**2)
    th_E_V2 = N0_sigma * (gamma_r/(1.0+gamma_r))**2 * N_t * G0 * (1.0 + abs_rho2)

    w_all = np.zeros((N_t, trials), dtype=complex)
    for k in range(trials):
        w_all[:,k:k+1], _ = beamforming_vec(h0, h1_hat_all[:,k:k+1])
    mc_inr = np.mean(np.abs((w_all.conj().T @ h1).reshape(-1))**2) * Tx_power / N0_vsat

    Dsq = (N_t**2) * abs_rho2 * G0 * G1 / ((1.0 + gamma_r)**2)
    EVsq = N0_sigma * (gamma_r / (1.0 + gamma_r))**2 * N_t * G0 * (1.0 + abs_rho2)
    inr_theory = (Tx_power / N0_vsat) * S_approx * (Dsq + EVsq)

    rows.append({
        "gamma": gamma,
        "G1": G1,
        "gamma_r": gamma_r,
        "E||hhat||^2/Nt (emp)": emp_hnorm,
        "E||hhat||^2/Nt (th)": th_hnorm,
        "Var z10 (emp)": emp_B10,
        "Var z10 (th)": th_B10,
        "E|rhohat|^2 (emp)": emp_E_rho2,
        "E|rhohat|^2 (th)": th_E_rho2,
        "S_def (empE[rho])": S_def,
        "S_approx (th)": S_approx,
        "E[h0* v v* h1] (emp)": lhs,
        "E[h0* v v* h1] (th)": rhs,
        "E|V|^2 (emp)": emp_E_V2,
        "E|V|^2 (th)": th_E_V2,
        "INR (MC)": mc_inr,
        "INR (theory)": inr_theory
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Plots
plt.figure()
plt.loglog(df["gamma"], df["E|rhohat|^2 (emp)"], marker="o", label="emp")
plt.loglog(df["gamma"], df["E|rhohat|^2 (th)"], marker="x", linestyle="--", label="th")
plt.xlabel("gamma")
plt.ylabel("E|rhohat|^2")
plt.title("Check of E|rhohat|^2")
plt.legend()
plt.grid(True, which="both")
plt.show()

plt.figure()
plt.loglog(df["gamma"], df["INR (MC)"], marker="o", label="MC")
plt.loglog(df["gamma"], df["INR (theory)"], marker="x", linestyle="--", label="theory")
plt.xlabel("gamma")
plt.ylabel("INR")
plt.title("INR: MC vs theory")
plt.legend()
plt.grid(True, which="both")
plt.show()
