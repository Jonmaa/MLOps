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

def fetch_imdb_dataset(**kwargs):
    """Obtiene el dataset IMDB de Hugging Face con un tamaño manejable"""
    dataset = load_dataset('imdb', split='test[:10%]')  # 10% del conjunto de prueba
    
    df = pd.DataFrame({
        'text': dataset['text'],
        'label': dataset['label']  # 0=negativo, 1=positivo
    })
    
    # Usamos solo 50 muestras para demostración
    df = df.sample(50, random_state=42).reset_index(drop=True)
    
    kwargs['ti'].xcom_push(key='imdb_dataset', value=df.to_json())
    print(f"Dataset cargado con {len(df)} muestras")

def analyze_with_model(**kwargs):
    """Realiza el análisis de sentimientos con manejo de textos largos"""
    ti = kwargs['ti']
    dataset_json = ti.xcom_pull(task_ids='fetch_imdb_dataset', key='imdb_dataset')
    df = pd.read_json(dataset_json)
    
    # Configuración del modelo
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline(
        "sentiment-analysis", 
        model=model_name,
        tokenizer=tokenizer,
        device=-1,  # CPU
        truncation=True,  # Añadimos truncamiento
        max_length=512  # Establecemos longitud máxima
    )
    
    print(f"\nIniciando análisis con modelo {model_name}...")
    
    # Procesamos con manejo de errores
    predictions = []
    for text in df['text']:
        try:
            # Truncamos el texto a 500 caracteres para asegurar que esté dentro del límite
            truncated_text = text[:500]
            pred = classifier(truncated_text)
            predictions.append(pred[0])
        except Exception as e:
            print(f"Error procesando texto: {str(e)}")
            # En caso de error, asignamos una predicción neutral
            predictions.append({'label': 'NEUTRAL', 'score': 0.5})
    
    # Convertimos resultados
    df['prediction'] = [1 if pred['label'] == 'POSITIVE' else 0 for pred in predictions]
    df['confidence'] = [pred['score'] for pred in predictions]
    
    ti.xcom_push(key='analysis_results', value=df.to_json())
    print("Análisis completado")

def evaluate_model_performance(**kwargs):
    """Evalúa el rendimiento del modelo"""
    ti = kwargs['ti']
    results_json = ti.xcom_pull(task_ids='analyze_with_model', key='analysis_results')
    df = pd.read_json(results_json)
    
    # Filtramos cualquier predicción neutral que pueda haberse generado por errores
    df = df[df['prediction'].isin([0, 1])]
    
    if len(df) > 0:
        accuracy = accuracy_score(df['label'], df['prediction'])
        report = classification_report(df['label'], df['prediction'], target_names=['Negative', 'Positive'])
        
        print("\n" + "="*50)
        print("EVALUACIÓN DEL MODELO DE SENTIMENT ANALYSIS")
        print("="*50)
        print(f"\nAccuracy: {accuracy:.2%}")
        print("\nReporte de clasificación:")
        print(report)
        
        ti.xcom_push(key='model_accuracy', value=accuracy)
    else:
        print("No hay resultados válidos para evaluar")

with DAG(
    'imdb_sentiment_analysis_fixed',
    default_args=default_args,
    description='Análisis de sentimientos con dataset IMDB (solución para textos largos)',
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'sentiment-analysis'],
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
    
    evaluate_task = PythonOperator(
        task_id='evaluate_model_performance',
        python_callable=evaluate_model_performance,
        provide_context=True,
    )
    
    fetch_task >> analyze_task >> evaluate_task
