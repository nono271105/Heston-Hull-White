# Simulation Monte Carlo pour Pricing d'Option Européenne sous Modèle Heston-Hull-White avec Sauts

Ce script calcule le prix d’un call européen via simulation Monte Carlo combinant :

* Modèle de volatilité stochastique **Heston**,
* Modèle de taux d’intérêt stochastique **Hull-White**,
* Sauts aléatoires sur le prix et la volatilité,
* Corrélations croisées entre sous-jacent, volatilité et taux.

---

## Fonctionnalités principales

* Simulations de trajectoires du sous-jacent, volatilité et taux sur 2 ans (252 jours par an).
* Intégration de sauts de Poisson sur prix et volatilité.
* Calcul du payoff actualisé pour estimer le prix de l’option.
* Estimation de l’erreur standard de Monte Carlo.

---

## Utilisation

1. Installer NumPy :

```bash
pip install numpy
```

2. Exécuter le script Python.

Le résultat affiche le prix moyen estimé du call et l’erreur standard.

---

## Paramètres clés

* `n_paths`: nombre de simulations (10 000 par défaut)
* `K`: strike de l’option
* Paramètres Heston (`kappa`, `theta`, `eta`), Hull-White (`a`, `sigma_r`), et sauts (`lambda_S`, `mu_J`, etc.)
* Corrélations modélisées via décomposition de Cholesky
