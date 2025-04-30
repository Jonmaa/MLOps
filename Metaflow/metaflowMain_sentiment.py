# pylint: skip-file

from metaflow import FlowSpec, step, card, current
from metaflow.cards import Markdown
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score
import pandas as pd


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

    
    @card(type="html")
    @step
    def predict(self):
        print("Ejecutando sentiment analysis...")

        texts = self.df_small.text.tolist()
        labels = self.df_small.label.tolist()
        preds, scores = [], []

        for txt in texts:
            out = self.classifier(txt[:512])[0]
            preds.append(1 if out['label']=='POSITIVE' else 0)
            scores.append(out['score'])

        acc = accuracy_score(labels, preds)
        current.card.append(Markdown(f"## 🔍 Accuracy: {acc*100:.2f}%"))

        self.next(self.end)

    @step
    def end(self):
        print("🎉 IMDB Sentiment analysis finalizado.")

if __name__ == "__main__":
    IMDBSentimentFlow()
