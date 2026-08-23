"""Deterministic ML training/evaluation core for MaintAI Studio (P0).

Pure Python + scikit-learn + xgboost. Fixed seeds, fixed small hyperparameters,
no MLflow, no LLM. Training returns JSON-safe Pydantic summaries separately from
runtime sklearn pipeline objects.
"""
