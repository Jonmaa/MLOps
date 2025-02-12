import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import numpy as np

# Cargar datos MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Definición del modelo
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

model = DeepNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Iniciar experimento en MLflow
mlflow.set_experiment("MNIST-Classification")

with mlflow.start_run():
    # Registrar hiperparámetros
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    # Entrenar modelo
    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("training_loss", avg_loss, step=epoch)
        print(f"🔄 Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Crear un ejemplo de entrada (un batch de imágenes aleatorias)
    example_input = torch.randn(1, 1, 28, 28)
    example_input_numpy = example_input.numpy()  # Convertirlo a numpy porque no acepta el formato torch.Tensor
    example_output = model(example_input) 
    
    # Generar la firma del modelo automáticamente
    signature = infer_signature(example_input.numpy(), example_output.detach().numpy())
    
    # Guardar el modelo con firma y ejemplo de entrada
    mlflow.pytorch.log_model(model, "mnist_model", signature=signature, input_example=example_input_numpy)

    # Guardar modelo localmente también
    torch.save(model.state_dict(), "modelo_entrenado.pth")

print("✅ Entrenamiento finalizado y modelo registrado en MLflow.")
