from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os

# Add the project root to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.full_search_pipeline import process_news_and_index

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "itmo_news_pipeline",
    default_args=default_args,
    description="Pipeline to parse and index ITMO news daily",
    schedule_interval="0 1 * * *",  # Run at 1 AM every day
    start_date=days_ago(1),
    catchup=False,
    tags=["itmo", "news", "embedding"],
)


def run_news_pipeline():
    """
    Wrapper function to run the news processing pipeline
    """
    process_news_and_index(
        url="https://news.itmo.ru/ru/main_news/", update_existing=True
    )


parse_and_index_task = PythonOperator(
    task_id="parse_and_index_itmo_news",
    python_callable=run_news_pipeline,
    dag=dag,
)

# Set task dependencies
parse_and_index_task
