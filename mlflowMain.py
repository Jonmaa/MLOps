import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import time
from mlflow.tracking import MlflowClient


# Definir wrapper PyFunc para servir el modelo
class PyTorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model.eval()

    def predict(self, context, model_input):
        with torch.no_grad():
            input_tensor = torch.tensor(model_input.values, dtype=torch.float32).view(-1, 1, 28, 28)
            output = self.model(input_tensor)
            return output.numpy()


# Datos
batch_size = 64
lr = 0.001
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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


class CNN_Model(nn.Module): # Modelo de red convolucional para mejorar los patrones de las imágenes
    def __init__(self):
        super(CNN_Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)  # Reduce el tamaño a la mitad
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = x.view(-1, 64 * 7 * 7)  # Aplanar antes de pasar a las capas densas
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# MLflow config
mlflow.set_tracking_uri("http://167.99.84.228:5000")
mlflow.set_experiment("MNIST-Classification")


with mlflow.start_run() as run:
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)

    # Entrenamiento
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

    # Ejemplo para schema
    example_input = torch.randn(4, 1, 28, 28)
    example_input_np = example_input.numpy()
    example_output = model(example_input).detach().numpy()

    # Convertir input a DataFrame (requerido por pyfunc)
    example_input_df = pd.DataFrame(example_input_np.reshape(4, -1))

    signature = infer_signature(example_input_df, example_output)
    print(f"✍️ Signature: {signature}")

    # Loguear modelo como pyfunc
    mlflow.pyfunc.log_model(
        artifact_path="mnist_model",
        python_model=PyTorchWrapper(model),
        input_example=example_input_df,
        signature=signature
    )

    # Métricas, tags y artefactos
    mlflow.log_metric("num_parameters", sum(p.numel() for p in model.parameters()))
    mlflow.set_tag("model_type", model.__class__.__name__)
    mlflow.set_tag("dataset", "MNIST")
    mlflow.set_tag("framework", "PyTorch")

    # Registrar en Model Registry
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/mnist_model",
        name="MNIST-Classifier"
    )

    time.sleep(10)
    client = MlflowClient()
    client.update_registered_model(
        name="MNIST-Classifier",
        description="Clasificador de dígitos MNIST con arquitecturas de redes neuronales usando PyTorch y PyFunc."
    )
    client.update_model_version(
        name="MNIST-Classifier",
        version=result.version,
        description=(
            f"Modelo entrenado con arquitectura {model.__class__.__name__} "
            f"y parámetros: learning rate={lr}, batch size={batch_size}, optimizador " + optimizer.__class__.__name__ 
        )
    )
    client.transition_model_version_stage(
        name="MNIST-Classifier",
        version=result.version,
        stage="Production"
    )


print("✅ Entrenamiento, logueo y registro finalizado.")

artifact_uri = mlflow.get_artifact_uri()
print(f"📂 Los artefactos se han guardado en: {artifact_uri}")