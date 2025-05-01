from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def fetch_imdb_dataset(ti, **kwargs):
    """
    1) Descarga TODO el split de test,
    2) Toma 250 ejemplos negativos y 250 positivos,
    3) Empuja el subset balanceado a XCom.
    """
    # 1) Descarga completo
    ds = load_dataset('imdb', split='test')
    df_full = pd.DataFrame({
        'text': ds['text'],
        'label': ds['label']  # 0=negativo, 1=positivo
    })

    # 2) Estratifica 250 de cada clase
    neg = df_full[df_full.label == 0].sample(250, random_state=42)
    pos = df_full[df_full.label == 1].sample(250, random_state=42)
    df_small = pd.concat([neg, pos]).reset_index(drop=True)

    # 3) Lo guardamos en XCom como JSON
    ti.xcom_push(key='imdb_dataset', value=df_small.to_json())
    print(f"✅ Subset IMDB listo: {len(df_small)} muestras (250 neg + 250 pos)")

def analyze_with_model(ti, **kwargs):
    """
    Recupera df_small de XCom, carga el modelo HuggingFace y evalúa.
    """
    # 1) Recupera el subset
    dataset_json = ti.xcom_pull(key='imdb_dataset', task_ids='fetch_imdb_dataset')
    df_small = pd.read_json(dataset_json)

    # 2) Inicializa el clasificador
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=tokenizer,
        device=-1,        # CPU
        truncation=True,
        max_length=512
    )

    texts  = df_small['text'].tolist()
    labels = df_small['label'].tolist()

    # 3) Predicciones
    preds, scores = [], []
    for txt in texts:
        out = classifier(txt[:512])[0]
        preds.append(1 if out['label']=='POSITIVE' else 0)
        scores.append(out['score'])

    # 4) Métricas
    acc    = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Negative','Positive'])

    print(f"\n🔍 Accuracy: {acc*100:.2f}%")
    print("\n🎯 Classification report:")
    print(report)

    # 5) Guardamos resultados para siguiente paso si hace falta
    results_df = pd.DataFrame({
        'text':    texts,
        'true':    labels,
        'pred':    preds,
        'score':   scores
    })
    ti.xcom_push(key='analysis_results', value=results_df.to_json())

# Definición del DAG
with DAG(
    'imdb_sentiment_analysis_fixed',
    default_args=default_args,
    description='Análisis de sentimientos IMDB con subset estratificado',
    schedule_interval=None,
    catchup=False,
    tags=['nlp','sentiment-analysis'],
) as dag:

    fetch_task = PythonOperator(
        task_id='fetch_imdb_dataset',
        python_callable=fetch_imdb_dataset,
        provide_context=True,
    )
    
    analyze_task = PythonOperator(
        task_id='analyze_with_model',
        python_callable=analyze_with_model,
        provide_context=True,
    )

    fetch_task >> analyze_task
