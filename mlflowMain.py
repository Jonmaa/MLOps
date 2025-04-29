import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import numpy as np
import time
import tempfile
import os
import uuid
import sys
from pathlib import Path
from mlflow.tracking import MlflowClient

# ===== CONFIGURACIÓN PARA EVITAR PROBLEMAS DE PERMISOS =====
# 1. Establecer variables de entorno para prevenir uso de directories protegidos
os.environ["MLFLOW_TRACKING_DIR"] = "/tmp/mlflow-tracking"
os.environ["MLFLOW_ARTIFACTS_DESTINATION"] = "/tmp/mlflow-artifacts"

# 2. Crear directorio local para artefactos y asegurar que existe
artifact_path = Path("/tmp/mlflow-artifacts")
artifact_path.mkdir(parents=True, exist_ok=True)

# 3. Imprimir información para diagnóstico
print(f"🔧 Directorio configurado para artefactos: {artifact_path}")
print(f"🔧 Usuario actual: {os.getuid()}")

# ===== CONFIGURACIÓN DEL MODELO Y DATOS =====
# Cargar datos MNIST
batch_size = 64
lr = 0.001
transform = transforms.Compose([transforms.ToTensor()])

try:
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("✅ Datos MNIST cargados correctamente")
except Exception as e:
    print(f"❌ Error al cargar datos MNIST: {e}")
    sys.exit(1)

# Definición del modelo - Usamos DeepNN que es más compleja y tiene mejor rendimiento
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

# ===== ENTRENAMIENTO Y REGISTRO DEL MODELO =====
try:
    model = DeepNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Configuración del servidor de seguimiento de MLflow
    mlflow.set_tracking_uri("http://167.99.84.228:5000") 
    print("🔌 Conectado a servidor MLflow")

    # Iniciar experimento
    mlflow.set_experiment("MNIST-Classification")
    print("📊 Experimento configurado")

    # Guardar modelo en local primero
    local_model_dir = Path("/tmp/mlflow-model-local")
    local_model_dir.mkdir(parents=True, exist_ok=True)
    
    with mlflow.start_run() as run:
        print(f"🏃 Iniciando ejecución MLflow: {run.info.run_id}")
        
        # Registrar hiperparámetros
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_type", "DeepNN")

        # Entrenamiento
        for epoch in range(5):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Calcular precisión durante entrenamiento
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
            
            # Métricas por época
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            # Registrar métricas
            mlflow.log_metric("training_loss", avg_loss, step=epoch)
            mlflow.log_metric("training_accuracy", accuracy, step=epoch)
            
            print(f"🔄 Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Crear ejemplo de entrada y salida para firma
        print("📝 Generando firma del modelo...")
        example_input = torch.randn(4, 1, 28, 28)
        example_output = model(example_input)
        
        # Guardar en archivo local primero
        local_model_file = local_model_dir / "model.pth"
        torch.save(model.state_dict(), local_model_file)
        print(f"💾 Modelo guardado localmente en: {local_model_file}")

        # Registrar modelo en MLflow
        print("📤 Subiendo modelo a MLflow...")
        
        # Firmar el modelo
        signature = infer_signature(
            example_input.numpy(), 
            example_output.detach().numpy()
        )
        
        # Guardar modelo en MLflow
        mlflow.pytorch.log_model(
            model, 
            "mnist_model",
            signature=signature,
            input_example=example_input.numpy(),
            registered_model_name="MNIST-Classifier"
        )
        
        print("✅ Modelo guardado en MLflow")
        
        # Obtener la última versión registrada
        client = MlflowClient()
        latest_version = client.get_latest_versions("MNIST-Classifier", stages=["None"])[0].version
        
        # Promocionar a producción
        print(f"🔼 Promocionando modelo a producción (versión {latest_version})...")
        client.transition_model_version_stage(
            name="MNIST-Classifier",
            version=latest_version,
            stage="Production"
        )
        
        print(f"🌟 Modelo promocionado a producción: MNIST-Classifier versión {latest_version}")
        
except Exception as e:
    print(f"❌ Error durante el entrenamiento o registro: {str(e)}")
    sys.exit(1)

print("✅ Proceso completado exitosamente")