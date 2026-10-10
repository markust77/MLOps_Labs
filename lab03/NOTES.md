# Lab 03 – Experiment Tracking & Hyperparameter Tuning: Notes

## Status
| Task | Status |
|---|---|
| I – MLflow Tracking (`mlflow_tracking.ipynb`) | in progress |
| II – Ray Tune (`ray_tune.ipynb`) | open |
| III – Diffusion model tuning (`diffusion.ipynb`) | open |

## Setup
- venv: `lab03/.venv` (Python 3.13, torch 2.14 cu130, mlflow 3.16, ray 2.58), CUDA verfügbar
- MLflow-Server im Verzeichnis `lab03` starten (dort liegen `mlflow.db` und `mlartifacts/`) → UI auf `localhost:8080`:
  ```bash
  cd ~/source/CAS/MLOps/MLOps_Labs/lab03
  source .venv/bin/activate
  mlflow server --host 127.0.0.1 --port 8080
  ```
- Notebook-Kernel: `lab03/.venv`- Backend: `mlflow.db` (SQLite) und `mlartifacts/` im Lab-Verzeichnis (beide in `.gitignore`)

## Task I – MLflow Tracking
Erledigt (02.10.):
- Experiment `Coffee_Models`: Prophet-Run mit Params, Metrics (MAPE, RMSE) und Modell geloggt
- Experiment `XOR`: erster Run mit `nn.Identity` als Aktivierung

Nächster Schritt: *Comparing Runs* – Run mit `F.sigmoid`, dann mit 100'000 Epochen, und die Loss-Kurven im Chart-Tab vergleichen.

### Learnings

## Task II – Ray Tune

### Learnings

## Task III – Diffusion model tuning

### Learnings
