# analysis.py
import math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0f1923'
plt.rcParams['axes.facecolor'] = '#1a2332'

# --------------------
# Physics functions
# --------------------
def reynolds(rho, U, L, mu):
    if mu <= 0 or L <= 0: return 0.0
    return (rho * U * L) / mu

def cd_blend(Re):
    if Re < 1: return 24 / max(Re, 1e-2)
    elif Re < 1e3: return 24/Re + 6/(1+math.sqrt(Re)) + 0.4
    else: return 0.44

def fin_eff(Re):
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp(-(x - 0.5) * 2.5))
    return max(0.02, min(1.0, t))

def flag_eff(Re):
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp((x - 0.0) * 2.5))
    return max(0.05, min(1.2, t * 1.2))

def thrust_fin(rho, L, freq, amp, Re):
    eff = fin_eff(Re)
    return 0.5 * rho * (freq * amp)**2 * L**2 * eff, eff

def thrust_flag(rho, mu, L, freq, pitch, Re):
    eff = flag_eff(Re)
    return 6 * math.pi * mu * (L/2) * (freq * pitch) * eff, eff

def drag_total(rho, mu, L, U):
    r = L / 2
    A = math.pi * r * r
    Re = reynolds(rho, abs(U), L, mu)
    Cd = cd_blend(max(Re, 1e-6))
    drag_linear = 6 * math.pi * mu * r * abs(U)
    drag_quad   = 0.5 * rho * Cd * A * U * U
    return drag_linear + drag_quad

def steady_speed(propulsion, rho, mu, L, freq):
    amp = max(1e-6, 0.2 * L)
    pitch = max(1e-6, 0.2 * L)
    U_lo, U_hi = 0.0, 3.0
    def net(U):
        Re = reynolds(rho, abs(U), L, mu)
        if propulsion == "Fin":
            T, _ = thrust_fin(rho, L, freq, amp, Re)
        else:
            T, _ = thrust_flag(rho, mu, L, freq, pitch, Re)
        return T - drag_total(rho, mu, L, U)
    if net(0.0) <= 0: return 0.0
    for _ in range(50):
        mid = 0.5*(U_lo+U_hi)
        if net(mid) > 0: U_lo = mid
        else: U_hi = mid
    return 0.5*(U_lo+U_hi)

# --------------------
# Main analysis function
# --------------------
def run_all_plots(outdir, scenario_name, L, freq, envs):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rho_baseline = 1000.0
    mu_values = np.logspace(-6, 1, 80)
    recs = []

    # --------------------
    # 1. Speed vs Viscosity
    # --------------------
    for prop in ["Fin", "Flagella"]:
        speeds = []
        for mu in mu_values:
            U = steady_speed(prop, rho_baseline, mu, L, freq)
            speeds.append(U)
            recs.append({"sweep":"mu","scenario":scenario_name,"propulsion":prop,
                         "rho":rho_baseline,"mu":mu,"L":L,"freq":freq,
                         "speed":U,"Re":reynolds(rho_baseline,U,L,mu)})
        fig = plt.figure()
        plt.loglog(mu_values, speeds)
        plt.xlabel("Viscosity μ (Pa·s)"); plt.ylabel("Steady speed U (m/s)")
        plt.title(f"{scenario_name.capitalize()} — {prop}: Speed vs Viscosity (ρ={rho_baseline})")
        plt.grid(True, which="major", alpha=0.3)
        fig.savefig(outdir/f"{scenario_name}_{prop}_speed_vs_mu.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # overlay + ratio
    fin = [steady_speed("Fin", rho_baseline, mu, L, freq) for mu in mu_values]
    flag= [steady_speed("Flagella", rho_baseline, mu, L, freq) for mu in mu_values]
    fig = plt.figure(); plt.loglog(mu_values, fin, label="Fin"); plt.loglog(mu_values, flag, label="Flagella")
    plt.xlabel("Viscosity μ (Pa·s)"); plt.ylabel("U (m/s)")
    plt.title(f"{scenario_name.capitalize()}: Fin vs Flagella — Speed vs Viscosity")
    plt.legend(); plt.grid(True, which="major", alpha=0.3)
    fig.savefig(outdir/f"{scenario_name}_Fin_vs_Flagella_speed_vs_mu.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    ratio = np.array(fin)/np.maximum(1e-12, np.array(flag))
    fig = plt.figure(); plt.semilogx(mu_values, ratio); plt.axhline(1.0, ls="--")
    plt.xlabel("Viscosity μ (Pa·s)"); plt.ylabel("Speed ratio (Fin/Flagella)")
    plt.title(f"{scenario_name.capitalize()}: Propulsion Advantage vs Viscosity")
    plt.grid(True, which="major", alpha=0.3)
    fig.savefig(outdir/f"{scenario_name}_advantage_ratio_vs_mu.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # --------------------
    # 2. Frequency sweeps per environment
    # --------------------
    for env_name, env in envs.items():
        rho, mu = env["rho"], env["mu"]
        freqs = np.logspace(-1, 3, 60) if scenario_name=="bacteria" else np.linspace(0.2, 10, 60)
        for prop in ["Fin","Flagella"]:
            speeds = []
            for f in freqs:
                U = steady_speed(prop, rho, mu, L, f)
                speeds.append(U)
                recs.append({"sweep":f"freq@{env_name}","scenario":scenario_name,"propulsion":prop,
                             "rho":rho,"mu":mu,"L":L,"freq":float(f),"speed":U,
                             "Re":reynolds(rho,U,L,mu)})
            fig = plt.figure(); plt.plot(freqs, speeds)
            plt.xlabel("Frequency f (Hz)"); plt.ylabel("U (m/s)")
            plt.title(f"{scenario_name.capitalize()} — {prop}: Speed vs Frequency in {env_name}")
            plt.grid(True, alpha=0.3)
            fig.savefig(outdir/f"{scenario_name}_{prop}_speed_vs_freq_{env_name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        # overlay
        speeds_fin  = [steady_speed("Fin", rho, mu, L, f) for f in freqs]
        speeds_flag = [steady_speed("Flagella", rho, mu, L, f) for f in freqs]
        fig = plt.figure(); plt.plot(freqs, speeds_fin, label="Fin"); plt.plot(freqs, speeds_flag, label="Flagella")
        plt.xlabel("Frequency f (Hz)"); plt.ylabel("U (m/s)")
        plt.title(f"{scenario_name.capitalize()} in {env_name}: Fin vs Flagella, Speed vs Frequency")
        plt.legend(); plt.grid(True, alpha=0.3)
        fig.savefig(outdir/f"{scenario_name}_Fin_vs_Flagella_speed_vs_freq_{env_name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # --------------------
    # 3. Size sweep (NEW)
    # --------------------
    sizes = np.logspace(-6, 1, 60)  # from micron to meter scale
    for env_name, env in envs.items():
        rho, mu = env["rho"], env["mu"]
        speeds_fin  = [steady_speed("Fin", rho, mu, Ls, freq) for Ls in sizes]
        speeds_flag = [steady_speed("Flagella", rho, mu, Ls, freq) for Ls in sizes]

        # save data
        for Ls, Uf, Ufla in zip(sizes, speeds_fin, speeds_flag):
            recs.append({"sweep":f"size@{env_name}","scenario":scenario_name,"propulsion":"Fin","rho":rho,"mu":mu,"L":Ls,"freq":freq,"speed":Uf,"Re":reynolds(rho,Uf,Ls,mu)})
            recs.append({"sweep":f"size@{env_name}","scenario":scenario_name,"propulsion":"Flagella","rho":rho,"mu":mu,"L":Ls,"freq":freq,"speed":Ufla,"Re":reynolds(rho,Ufla,Ls,mu)})

        fig = plt.figure()
        plt.loglog(sizes, speeds_fin, label="Fin")
        plt.loglog(sizes, speeds_flag, label="Flagella")
        plt.xlabel("Organism Size L (m)")
        plt.ylabel("Steady Speed U (m/s)")
        plt.title(f"{scenario_name.capitalize()} in {env_name}: Speed vs Size")
        plt.legend(); plt.grid(True, which="major", alpha=0.3)
        fig.savefig(outdir/f"{scenario_name}_Fin_vs_Flagella_speed_vs_size_{env_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --------------------
    # Save all data
    # --------------------
    pd.DataFrame(recs).to_csv(outdir/f"{scenario_name}_analysis.csv", index=False)
