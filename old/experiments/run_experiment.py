def main(config_path):
    config = load_config(config_path)

    mlflow.set_experiment(config["experiment"]["name"])
    with mlflow.start_run():
        log_config(config)
        metrics = run_pipeline(config)
        mlflow.log_metrics(metrics)