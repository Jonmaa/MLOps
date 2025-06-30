import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from transformers import pipeline, AutoTokenizer
import pandas as pd
import time
from mlflow.tracking import MlflowClient

class SentimentWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_context(self, context):
        from transformers import pipeline, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=-1,
            truncation=True,
            max_length=512
        )

    def predict(self, context, model_input: pd.DataFrame):
        texts = model_input["text"].tolist()
        preds = self.classifier(texts)
        return pd.DataFrame(preds)



mlflow.set_tracking_uri("http://167.99.84.228:5000")
mlflow.set_experiment("IMDB-Sentiment")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"


with mlflow.start_run() as run:
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("max_length", 512)

    # 2) Prepara un ejemplo pequeño para la signature
    example_texts = ["I love this!", "This is awful..."]
    example_df = pd.DataFrame({"text": example_texts})

    # Ejecuta el pipeline para obtener output example
    classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    example_out = classifier(example_texts)
    example_out_df = pd.DataFrame(example_out)
    sig = infer_signature(example_df, example_out_df)
    mlflow.log_param("signature", str(sig))

    # 3) Loguea el modelo PyFunc
    mlflow.pyfunc.log_model(
        artifact_path="sentiment_model",
        python_model=SentimentWrapper(model_name),
        input_example=example_df,
        signature=sig
    )


    # 4) Registra en Model Registry
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/sentiment_model",
        name="IMDB-Sentiment-Classifier"
    )

    time.sleep(10)
    client = MlflowClient()

    client.transition_model_version_stage(
        name="IMDB-Sentiment-Classifier",
        version=result.version,
        stage="Production"
    )

print("✅ Modelo de sentimiento registrado en MLflow")
