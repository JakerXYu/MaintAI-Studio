-- Create the MLflow tracking backend database on first Postgres init.
-- The `maintai` database is created automatically via POSTGRES_DB.
CREATE DATABASE mlflow;
