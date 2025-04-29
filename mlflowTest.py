import torch
from torchvision import datasets, transforms
import mlflow
import mlflow.pyfunc
import pandas as pd
import torch.nn as nn

# Conectar con el servidor MLflow
mlflow.set_tracking_uri("http://167.99.84.228:5000")
mlflow.set_experiment("MNIST-Classification")

# Cargar el modelo registrado como pyfunc
model = mlflow.pyfunc.load_model("models:/MNIST-Classifier/Production")
print("🔄 Modelo cargado desde MLflow Registry.")

# Preparar datos de test
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluar modelo
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
test_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        # Convertir imágenes a DataFrame como espera pyfunc
        images_np = images.numpy().reshape(images.shape[0], -1)
        images_df = pd.DataFrame(images_np)

        outputs = model.predict(images_df)
        outputs_tensor = torch.tensor(outputs)
        
        loss = criterion(outputs_tensor, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs_tensor, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
test_loss /= len(test_loader)

print(f"🔍 Precisión en test: {accuracy:.2f}%")
print(f"🎯 Pérdida en test: {test_loss:.4f}")

# Registrar métricas
with mlflow.start_run():
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", test_loss)

    from sklearn.metrics import classification_report
    report = classification_report(labels.numpy(), predicted.numpy(), output_dict=True)
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):  # Ignorar las métricas globales
            mlflow.log_metric(f"precision_class_{class_label}", metrics["precision"])
            mlflow.log_metric(f"recall_class_{class_label}", metrics["recall"])
            mlflow.log_metric(f"f1_score_class_{class_label}", metrics["f1-score"])

# Validación automática
if accuracy < 90:
    print("❌ Precisión demasiado baja. Fallando el workflow.")
    # exit(1)
else:
    print("✅ Test aprobado. Modelo validado.")
