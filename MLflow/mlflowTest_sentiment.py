import mlflow
import mlflow.pyfunc
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report


# Conectar con el servidor MLflow
mlflow.set_tracking_uri("http://167.99.84.228:5000")
mlflow.set_experiment("IMDB-Sentiment")

# 1) Carga el modelo en producción
model = mlflow.pyfunc.load_model("models:/IMDB-Sentiment-Classifier/Production")
print("🔄 Modelo de sentimiento cargado desde MLflow Registry.")

# 2) Cargar dataset IMDB (10% del test para que sea manejable)
ds = load_dataset("imdb", split="test[:10%]")
df = pd.DataFrame({
    "text": ds["text"],
    "label": ds["label"]  # 0 = negative, 1 = positive
})

# Opcional: tomar solo N muestras si quieres un subset más pequeño
# 250 negativos + 250 positivos
neg = df[df.label == 0].sample(250, random_state=42)
pos = df[df.label == 1].sample(250, random_state=42)
df_small = pd.concat([neg, pos]).reset_index(drop=True)

# 3) Predecir con el modelo PyFunc
# El wrapper devolvió un DataFrame con columnas ['label','score']
preds_df = model.predict(df_small[["text"]])

# Mapear etiquetas de texto a 0/1
df["pred"] = preds_df["label"].map({"NEGATIVE": 0, "POSITIVE": 1})
df["score"] = preds_df["score"]

# 4) Evaluar métricas
accuracy = accuracy_score(df["label"], df["pred"])
report = classification_report(
    df["label"], 
    df["pred"], 
    target_names=["Negative", "Positive"], 
    output_dict=True
)

print(f"\n🔍 Accuracy on IMDB subset: {accuracy:.2%}")
print("\n🎯 Classification report:")
print(classification_report(df["label"], df["pred"], target_names=["Negative", "Positive"]))

# 5) Loguear métricas en MLflow
with mlflow.start_run():
    mlflow.log_metric("test_accuracy", accuracy)
    # Registrar precision/recall/f1 por clase
    for cls in ["Negative", "Positive"]:
        mlflow.log_metric(f"precision_{cls}", report[cls]["precision"])
        mlflow.log_metric(f"recall_{cls}",    report[cls]["recall"])
        mlflow.log_metric(f"f1_{cls}",        report[cls]["f1-score"])

# Validación automática (opcional)
if accuracy < 0.80:
    print("❌ Accuracy por debajo del umbral (80%).")
else:
    print("✅ Test de sentimiento aprobado.")
