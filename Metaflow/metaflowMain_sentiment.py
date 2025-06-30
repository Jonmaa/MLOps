# pylint: skip-file

from metaflow import FlowSpec, step, card, current
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score
import pandas as pd
import io
import numpy as np
import torch
import base64
from sklearn.manifold import TSNE
from metaflow.cards import Markdown, Image



class IMDBSentimentFlow(FlowSpec):

    @step
    def start(self):
        print("Descargando el dataset...")

        ds = load_dataset("imdb", split="test")
        df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})

        neg = df[df.label == 0].sample(250, random_state=42)
        pos = df[df.label == 1].sample(250, random_state=42)

        self.df_small = pd.concat([neg, pos]).reset_index(drop=True)

        self.next(self.load_model)

    @step
    def load_model(self):
        print("Cargando el modelo...")

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
            device=-1
        )

        self.next(self.predict)

    
    @card
    @step
    def predict(self):
        print("Ejecutando sentiment analysis...")

        texts = self.df_small.text.tolist()
        labels = self.df_small.label.tolist()
        preds, scores = [], []
        embeddings = []

        for txt in texts:
            # Tokenización manual para acceder a embeddings
            inputs = self.tokenizer(txt[:512], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.classifier.model(**inputs, output_hidden_states=True)

            # Usamos la salida de la penúltima capa oculta del CLS token
            last_hidden = outputs.hidden_states[-2][0][0].numpy()
            embeddings.append(last_hidden)

            # Predicción de etiqueta
            out = self.classifier(txt[:512])[0]
            preds.append(1 if out['label'] == 'POSITIVE' else 0)
            scores.append(out['score'])

        # Calcular accuracy
        acc = accuracy_score(labels, preds)
        print(f"Accuracy: {acc*100:.2f}%")
        current.card.append(Markdown(f"## 🔍 Accuracy: {acc*100:.2f}%"))

        # t-SNE
        print("Reduciendo dimensiones con t-SNE...")
        X = np.vstack(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_reduced = tsne.fit_transform(X)

        # Graficar
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
        legend = ax.legend(*scatter.legend_elements(), title="Etiqueta real")
        ax.add_artist(legend)
        ax.set_title("t-SNE de representaciones del modelo")

        # Guardar gráfico como base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        current.card.append(Markdown(f"![t-SNE](data:image/png;base64,{img_b64})"))
        plt.close(fig)

        self.next(self.end)


    @step
    def end(self):
        print("🎉 IMDB Sentiment analysis finalizado.")
        self.classifier.model.save_pretrained("sentiment_model/")
        self.tokenizer.save_pretrained("sentiment_model/")

if __name__ == "__main__":
    IMDBSentimentFlow()