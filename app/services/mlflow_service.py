import mlflow


class MLflowService:
    def __init__(self):

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Image Classification API")

    def log_prediction(
        self,
        model_name: str,
        device: str,
        confidence: float,
        inference_time_ms: float,
        predicted_class: str,
    ):

        with mlflow.start_run():
            mlflow.log_param("model_name", model_name)

            mlflow.log_param("device", device)

            mlflow.log_param("predicted_class", predicted_class)

            mlflow.log_metric("confidence", confidence)

            mlflow.log_metric("inference_time_ms", inference_time_ms)
