from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
from sklearn.metrics import accuracy_score, classification_report

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
    
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = x.view(-1, 28*28) 
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
    
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

# --- Parámetros por defecto del DAG ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Tareas del DAG
def train_model(**kwargs):

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='/tmp/mnist_data', train=True,
                              transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64,
                                               shuffle=True)

    model = DeepNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


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


    model_path = '/tmp/mnist_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {model_path}")

    ti = kwargs['ti']
    ti.xcom_push(key='model_path', value=model_path)


def evaluate_model(**kwargs):

    ti = kwargs['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root='/tmp/mnist_data', train=False,
                             transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64,
                                              shuffle=False)


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

    t1 >> t2
