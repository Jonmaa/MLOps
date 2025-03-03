from metaflow import FlowSpec, step, Parameter, IncludeFile, card, current
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import io
from sklearn.metrics import confusion_matrix
from PIL import Image

# Definición del modelo CNN
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
    lr = Parameter('learning_rate', default=0.01, help='Tasa de aprendizaje')
    epochs = Parameter('epochs', default=5, help='Número de épocas')
    
    @step
    def start(self):
        print("📌 Inicio del flujo de entrenamiento MNIST")
        self.next(self.load_data)
    
    @step
    def load_data(self):
        print("📥 Cargando datos MNIST...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.next(self.train_model)
    
    @step
    def train_model(self):
        print("🚀 Entrenando modelo...")
        self.model = SimpleNN()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"🔄 Epoch {epoch+1}, Loss: {total_loss / len(self.train_loader):.4f}")
        self.next(self.save_model)
    
    @step
    def save_model(self):
        print("💾 Guardando modelo entrenado...")
        with open("modelo_entrenado.pkl", "wb") as f:
            pickle.dump(self.model, f)
        print("✅ Modelo guardado correctamente.")
        self.next(self.evaluate_model)
    
    @card
    @step
    def evaluate_model(self):
       print("📊 Evaluando el modelo...")
       self.model.eval()
       correct = 0
       total = 0
       all_preds = []
       all_labels = []
       test_images = []

       with torch.no_grad():
           for images, labels in self.test_loader:
               outputs = self.model(images)
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
               all_preds.extend(predicted.numpy())
               all_labels.extend(labels.numpy())
               test_images.extend(images.numpy())

       accuracy = correct / total
       print(f"✅ Precisión del modelo: {accuracy * 100:.2f}%")

       # Seleccionar 10 imágenes del test
       num_samples = 10
       sample_images = test_images[:num_samples]
       sample_labels = all_labels[:num_samples]
       sample_preds = all_preds[:num_samples]

       # Crear la figura
       fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
       for i, ax in enumerate(axes):
           ax.imshow(sample_images[i].squeeze(), cmap='gray')
           ax.set_title(f"Pred: {sample_preds[i]}\nReal: {sample_labels[i]}")
           ax.axis("off")

       # Guardar la imagen en un buffer
       buffer = io.BytesIO()
       plt.savefig(buffer, format='png')
       buffer.seek(0)
       plt.close()

       # Convertir la imagen en Base64
       import base64
       img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

       # Agregar la imagen a la card de Metaflow
       current.card.append(f'<h3>🔍 Ejemplos de predicciones</h3>')
       current.card.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')

       print("📸 Predicciones visualizadas en la card de Metaflow")
       self.next(self.end)

    @step
    def end(self):
        print("🎉 Flujo de trabajo completado.")

if __name__ == '__main__':
    MNISTFlow()
