from mlflow.tracking import MlflowClient

client = MlflowClient()

for model in client.search_registered_models():
    print("=" * 50)
    print(f"Model Name: {model.name}")

    for version in model.latest_versions:
        print(
            f"Version: {version.version}, "
            f"Stage: {version.current_stage}"
        )