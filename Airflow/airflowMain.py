from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
from sklearn.metrics import accuracy_score, classification_report

# --- Definición del modelo (DeepNN idéntico al de Metaflow) ---
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
        x = torch.relu(self.fc1(x)); x = self.dropout(x)
        x = torch.relu(self.fc2(x)); x = self.dropout(x)
        x = torch.relu(self.fc3(x)); x = self.dropout(x)
        x = torch.relu(self.fc4(x)); x = self.dropout(x)
        return self.fc5(x)

# --- Parámetros por defecto del DAG ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- Funciones de las tareas ---
def train_model(**kwargs):
    # 1) Crea DataLoader de entrenamiento
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='/tmp/mnist_data', train=True,
                              transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64,
                                               shuffle=True)
    # 2) Instancia el modelo y optimizador
    model = DeepNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3) Bucle de entrenamiento
    for epoch in range(10):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10, loss={total_loss/len(train_loader):.4f}")

    # 4) Serializa el modelo a disco
    model_path = '/tmp/mnist_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {model_path}")

    # 5) Comunica la ruta del modelo al siguiente paso
    ti = kwargs['ti']
    ti.xcom_push(key='model_path', value=model_path)


def evaluate_model(**kwargs):
    # 1) Recupera la ruta del modelo desde XCom
    ti = kwargs['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    # 2) Carga el modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model.eval()

    # 3) Prepara DataLoader de test
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root='/tmp/mnist_data', train=False,
                             transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64,
                                              shuffle=False)

    # 4) Inferencia y métricas
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🔍 Accuracy en test: {acc*100:.2f}%\n")
    print(classification_report(all_labels, all_preds,
                                target_names=[str(i) for i in range(10)]))

    # 5) Opcional: push de la métrica
    ti.xcom_push(key='test_accuracy', value=acc)


# --- Definición del DAG ---
with DAG(
    dag_id='mnist_classification_airflow',
    default_args=default_args,
    description='Entrena y evalúa MNIST con PyTorch (DeepNN)',
    schedule_interval=None,
    catchup=False,
    tags=['mnist','pytorch'],
) as dag:

    t1 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )

    # Orden de ejecución
    t1 >> t2
