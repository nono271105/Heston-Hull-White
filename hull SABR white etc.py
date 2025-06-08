import numpy as np

# === Paramètres ===
T = 2.0
N = 252
dt = T / N
n_paths = 10000  # Nombre de simulations Monte Carlo
K = 100          # Strike de l'option

# === Paramètres initiaux ===
S0 = 100
v0 = 0.04
r0 = 0.03

# === Heston params ===
kappa = 2.0
theta = 0.04
eta = 0.2

# === Hull-White params ===
a = 0.1
sigma_r = 0.01
b = lambda t: 0.03

# === Jumps params ===
lambda_S = 0.2
mu_J = -0.1
sigma_J = 0.2

lambda_v = 0.1
mu_Jv = 0.0
sigma_Jv = 0.1

# === Corrélations ===
rho_Sv = -0.5
rho_Sr = 0.2
rho_vr = 0.1

# Matrice de corrélation
corr_matrix = np.array([
    [1.0,     rho_Sv, rho_Sr],
    [rho_Sv,  1.0,    rho_vr],
    [rho_Sr,  rho_vr, 1.0]
])
L = np.linalg.cholesky(corr_matrix)

# === Simulation Monte Carlo ===
payoffs = []

np.random.seed(42)

for _ in range(n_paths):
    S = S0
    v = v0
    r = r0
    r_avg = 0  # intégrale de r_t sur [0,T]

    for t in range(N):
        Z = np.random.normal(size=3)
        dW = L @ (np.sqrt(dt) * Z)

        # Sauts
        jump_S = np.sum(np.random.normal(mu_J, sigma_J, np.random.poisson(lambda_S * dt)))
        jump_v = np.sum(np.random.normal(mu_Jv, sigma_Jv, np.random.poisson(lambda_v * dt)))

        # Volatilité
        v = v + kappa * (theta - v) * dt + eta * np.sqrt(max(v, 0)) * dW[1] + jump_v
        v = max(v, 1e-6)

        # Taux
        r = r + a * (b(t * dt) - r) * dt + sigma_r * dW[2]
        r_avg += r * dt

        # Prix sous-jacent
        S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v) * dW[0] + jump_S)

    # Payoff actualisé
    payoff = np.exp(-r_avg) * max(S - K, 0)
    payoffs.append(payoff)

# === Prix estimé de l'option ===
call_price = np.mean(payoffs)
std_error = np.std(payoffs) / np.sqrt(n_paths)

print(f"Prix estimé du call européen : {call_price:.4f}")
print(f"Erreur standard Monte Carlo : {std_error:.4f}")