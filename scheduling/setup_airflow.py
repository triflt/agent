import os
from pathlib import Path


def setup_airflow():
    airflow_home = Path(__file__).parent / "airflow_home"
    os.environ["AIRFLOW_HOME"] = str(airflow_home)

    dags_dir = airflow_home / "dags"
    logs_dir = airflow_home / "logs"
    plugins_dir = airflow_home / "plugins"

    for dir_path in [airflow_home, dags_dir, logs_dir, plugins_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    if not (dags_dir / "itmo_news_dag.py").exists():
        os.symlink(
            Path(__file__).parent / "dags" / "itmo_news_dag.py",
            dags_dir / "itmo_news_dag.py",
        )

    print(f"Airflow home directory set to: {airflow_home}")
    print("To initialize Airflow database, run:")
    print("airflow db init")
    print("\nTo create an admin user, run:")
    print(
        "airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com"
    )
    print("\nTo start the webserver (in a separate terminal), run:")
    print("airflow webserver --port 8080")
    print("\nTo start the scheduler (in a separate terminal), run:")
    print("airflow scheduler")


if __name__ == "__main__":
    setup_airflow()
