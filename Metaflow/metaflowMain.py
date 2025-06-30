from metaflow import FlowSpec, step, Parameter, card, current
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import matplotlib.pyplot as plt
import random
import io, base64
from sklearn.manifold import TSNE
import numpy as np
from metaflow.cards import Markdown

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10) # 10 salidas diferentes, una por cada dígito (0-9)

    def forward(self, x):
        x = x.view(-1, 28*28) # Aplanar la imagen a un vector de 784 valores
        x = torch.relu(self.fc1(x)) # Relu como función de activación
        x = self.fc2(x)
        return x

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class MNISTFlow(FlowSpec):
    batch_size = Parameter('batch_size', default=64, help='Tamaño del batch')
    lr = Parameter('learning_rate', default=0.001, help='Tasa de aprendizaje')
    epochs = Parameter('epochs', default=10, help='Número de épocas')

    @step
    def start(self):
        print("Inicio del flujo MNISTFlow")
        self.next(self.load_data)

    @step
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(root='data', train=True, transform=transform, download=True)
        test_ds = datasets.MNIST(root='data', train=False, transform=transform, download=True)
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Entrenando modelo...")
        self.model = DeepNN()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward(); optimizer.step(); total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, loss = {total_loss/len(self.train_loader):.4f}")
        self.next(self.save_model)

    @step
    def save_model(self):
        with open('model.pkl','wb') as f: pickle.dump(self.model, f)
        print("Modelo serializado en model.pkl")
        self.next(self.evaluate_model)

    @card
    @step
    def evaluate_model(self):
        print("Evaluación en test set...")
        self.model.eval()
        all_preds, all_labels, all_imgs = [], [], []
        embeddings = []
    
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds += preds.tolist()
                all_labels += labels.tolist()
                all_imgs += images.numpy().tolist()
                # Obtener embeddings del penúltimo layer
                x = images.view(-1, 28 * 28)
                layer_output = torch.relu(self.model.fc4(torch.relu(self.model.fc3(torch.relu(self.model.fc2(torch.relu(self.model.fc1(x))))))))
                embeddings.append(layer_output.numpy())
    
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        print(f"Accuracy: {acc*100:.2f}%")
        current.card.append(Markdown(f"## 🔍 Accuracy: {acc*100:.2f}%"))
    
        # Reducimos los embeddings a 2D usando t-SNE
        embeddings = np.vstack(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(embeddings)
    
        # Graficamos la dispersión
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
        legend = ax.legend(*scatter.legend_elements(), title="Clases")
        ax.add_artist(legend)
        ax.set_title("Visualización t-SNE de las representaciones aprendidas")
        
        # Guardamos la figura como base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        current.card.append(Markdown(f"![t-SNE](data:image/png;base64,{img_b64})"))
        plt.close(fig)
    
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Run completo")

if __name__=='__main__': MNISTFlow()