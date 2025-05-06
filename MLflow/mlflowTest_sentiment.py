import mlflow
import mlflow.pyfunc
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    # Conectar con el servidor MLflow
    mlflow.set_tracking_uri("http://167.99.84.228:5000")
    mlflow.set_experiment("IMDB-Sentiment")

    # 1) Carga el modelo en producción
    model = mlflow.pyfunc.load_model(
        "models:/IMDB-Sentiment-Classifier/Production"
    )
    print("🔄 Modelo de sentimiento cargado desde MLflow Registry.")

    # 2) Cargar todo el dataset IMDB porque si cojo solo el 10% me saca todo negativos
    ds = load_dataset("imdb", split="test")
    df_full = pd.DataFrame({
        "text": ds["text"],
        "label": ds["label"]  # 0=negativo, 1=positivo
    })

    # 3) Estratificar: 250 negativos + 250 positivos
    neg = df_full[df_full.label == 0].sample(250, random_state=42)
    pos = df_full[df_full.label == 1].sample(250, random_state=42)
    df_small = pd.concat([neg, pos]).reset_index(drop=True)

    # 4) Predecir con el modelo PyFunc
    preds_df = model.predict(df_small[["text"]])  # columnas ['label','score']

    # 5) Mapear etiquetas de texto a 0/1 y unir al df_small
    df_small["pred"] = preds_df["label"].map({"NEGATIVE": 0, "POSITIVE": 1})
    df_small["score"] = preds_df["score"]

    # 6) Calcular métricas sobre df_small
    accuracy = accuracy_score(df_small["label"], df_small["pred"])
    report_dict = classification_report(
        df_small["label"],
        df_small["pred"],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0
    )

    print(f"\n🔍 Accuracy on IMDB subset: {accuracy:.2%}")
    print("\n🎯 Classification report:")
    print(classification_report(
        df_small["label"],
        df_small["pred"],
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    # 7) Loguear métricas en MLflow
    with mlflow.start_run():
        mlflow.log_metric("test_accuracy", accuracy)
        for cls in ["Negative", "Positive"]:
            mlflow.log_metric(f"precision_{cls}", report_dict[cls]["precision"])
            mlflow.log_metric(f"recall_{cls}",    report_dict[cls]["recall"])
            mlflow.log_metric(f"f1_{cls}",        report_dict[cls]["f1-score"])

    # 8) Validación automática
    if accuracy < 0.80:
        print("❌ Accuracy por debajo del umbral (80%).")
    else:
        print("✅ Test de sentimiento aprobado.")
