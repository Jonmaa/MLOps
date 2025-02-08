import torch
import torch.nn as nn
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
from main import SimpleNN  # Importamos el modelo

# Cargar datos de test MNIST
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cargar el modelo entrenado desde MLflow
model = SimpleNN()
model.load_state_dict(torch.load("modelo_entrenado.pth"))
model.eval()

# Evaluación del modelo
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
test_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
test_loss /= len(test_loader)

print(f"🔍 Precisión en test: {accuracy:.2f}%")
print(f"🎯 Pérdida en test: {test_loss:.4f}")

# Registrar métricas en MLflow
mlflow.set_experiment("MNIST-Classification")

with mlflow.start_run():
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", test_loss)

# Falla si la precisión es menor al 90%
if accuracy < 90:
    print("❌ Precisión demasiado baja. Fallando el workflow.")
    exit(1)  # Detiene GitHub Actions
else:
    print("✅ Test aprobado. Modelo validado.")
