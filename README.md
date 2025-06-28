# ETD Data Assimilation

A sandbox for exploring (stochastic)**Exponential Time Differencing (ETD)** and **Stochastic Integrating Factor (SIF)** methods in the context of **data assimilation** using **particle filters** and **Kalman filters**.

This repository serves as a flexible platform for testing time-stepping schemes, stochastic integration strategies, and data assimilation algorithms for applications in moderately-high-dimensions or stiff systems.

## [📘 See Detailed Examples →](EXAMPLES.md)

---

## ✨ Features
- 🧠 Autodifferentiable via JAX — supports gradient-based optimization and learning
- ⚙️**GPU and CPU compatible** — tested on NVIDIA GPUs and standard CPUs
- 🌐 Spectral spatial discretisation
- ⏱️ Timestepping: 
  - **Runge-Kutta(RK) and Stochastic Runge-Kutta**
  - **Exponential time differencing (ETDRK) and Stochastic  (ETDRK)**
  - **Integrating Factor Runge Kutta(IFRK) and Stochastic (IFRK)**

- 📈  Data Assimilation:
  - **Particle Filters (PF)**
  - **Ensemble Kalman Filter(EnKF)**

- 🔧 Tools for:
  - Abstract class for development of filtering with sparse data
  - Synthetic data generation
  - Convergence testing
  - Visualization of ensemble trajectories and filter performance
  - Configurations for reproducible experiments


## 📁 Repository Structure

```bash
etd-data-assimilation/
├── filters/           # Particle & Kalman filtering modules
├── models/            # ETD, SIF, and other time-stepping schemes
├── models/ETD_KT_CM_JAX_Vecotrised.py # Dynamical systems (e.g. KS, KdV, SPDEs)
├── tests/             # unit tests,
├── Saving/            # Images generated
├── metrics/           # Some utility functions
├── examples/          #Examples of the usage
└── README.md          # You're here! Overview of examples
```

examples/

[📘 See Detailed Examples →](EXAMPLES.md)
