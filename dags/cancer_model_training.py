from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="train_and_register_cancer_model",
    description="Entrena modelos y registra el de mejor ROC AUC en MLflow.",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["mlflow", "training"],
) as dag:
    train_models = BashOperator(
        task_id="train_models",
        bash_command="python /opt/airflow/project/scripts/train.py",
    )

    register_champion = BashOperator(
        task_id="register_champion",
        bash_command="python /opt/airflow/project/scripts/register_champion.py",
    )

    train_models >> register_champion
