import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from typing import NamedTuple
import os

KFP_VERSION = '1.8.9'

def fetch_imdb_dataset(output_dir: str = '/mnt/shared/imdb_data') -> NamedTuple('Outputs', [('dataset_path', str)]):
    from datasets import load_dataset
    import pandas as pd
    import os
    from collections import namedtuple
    
    dataset = load_dataset('imdb', split='test[:10%]')
    df = pd.DataFrame({'text': dataset['text'], 'label': dataset['label']})
    df = df.sample(50, random_state=42).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'imdb_dataset.csv')
    df.to_csv(dataset_path, index=False)

    if os.path.exists(dataset_path):
        print(f"El archivo se ha guardado correctamente en: {dataset_path}")
    else:
        print(f"Error: El archivo no se ha guardado en la ruta {dataset_path}")

    outputs = namedtuple('Outputs', ['dataset_path'])
    return outputs(dataset_path)

fetch_imdb_dataset_op = func_to_container_op(
    fetch_imdb_dataset,
    base_image='python:3.9-slim',
    packages_to_install=['datasets', 'pandas']
)

def analyze_with_model(dataset_path: str, output_dir: str = '/mnt/shared/analysis_results') -> NamedTuple('Outputs', [('results_path', str)]):
    import pandas as pd
    from transformers import pipeline, AutoTokenizer
    import os
    from collections import namedtuple
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, device=-1, truncation=True, max_length=512)

    predictions = []
    for text in df['text']:
        try:
            truncated_text = text[:500]
            pred = classifier(truncated_text)
            predictions.append(pred[0])
        except Exception:
            predictions.append({'label': 'NEUTRAL', 'score': 0.5})
    
    df['prediction'] = [1 if pred['label'] == 'POSITIVE' else 0 for pred in predictions]
    df['confidence'] = [pred['score'] for pred in predictions]
    
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'analysis_results.csv')
    df.to_csv(results_path, index=False)
    
    outputs = namedtuple('Outputs', ['results_path'])
    return outputs(results_path)

analyze_with_model_op = func_to_container_op(
    analyze_with_model,
    base_image='python:3.9-slim',
    packages_to_install=['transformers', 'pandas', 'torch']
)

def evaluate_model_performance(results_path: str) -> NamedTuple('Outputs', [('accuracy', float)]):
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    from collections import namedtuple
    
    df = pd.read_csv(results_path)
    df = df[df['prediction'].isin([0, 1])]

    if len(df) > 0:
        accuracy = accuracy_score(df['label'], df['prediction'])
        print(f"\nAccuracy: {accuracy:.2%}")
        print(classification_report(df['label'], df['prediction'], target_names=['Negative', 'Positive']))
    else:
        accuracy = 0.0
    
    outputs = namedtuple('Outputs', ['accuracy'])
    return outputs(accuracy)

evaluate_model_performance_op = func_to_container_op(
    evaluate_model_performance,
    base_image='python:3.9-slim',
    packages_to_install=['pandas', 'scikit-learn']
)

@dsl.pipeline(
    name='IMDB Sentiment Analysis Pipeline',
    description='Un pipeline para análisis de sentimientos en reviews de IMDB usando Hugging Face'
)
def imdb_sentiment_pipeline():
    # Crear un volumen persistente
    volume_op = dsl.VolumeOp(
        name="create-pvc",
        resource_name="imdb-pvc",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWO
    )

    # Etapa 1: Descargar el dataset
    fetch_task = fetch_imdb_dataset_op(output_dir='/mnt/shared/imdb_data').add_pvolumes({"/mnt/shared": volume_op.volume})

    # Etapa 2: Analizar el dataset
    analyze_task = analyze_with_model_op(
        dataset_path=fetch_task.outputs['dataset_path'],
        output_dir='/mnt/shared/analysis_results'
    ).add_pvolumes({"/mnt/shared": fetch_task.pvolume})

    # Etapa 3: Evaluar el modelo
    evaluate_task = evaluate_model_performance_op(
        results_path=analyze_task.outputs['results_path']
    ).add_pvolumes({"/mnt/shared": analyze_task.pvolume})

def compile_pipeline():
    import kfp.compiler as compiler
    pipeline_filename = 'imdb_sentiment_analysis_pipeline.yaml'
    compiler.Compiler().compile(imdb_sentiment_pipeline, pipeline_filename)
    print(f"Pipeline compilado como {pipeline_filename}")

if __name__ == '__main__':
    compile_pipeline()