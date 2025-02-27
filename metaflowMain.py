from metaflow import FlowSpec, step, Parameter, current
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import pickle  # Using pickle instead of joblib

class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST digit classification.
    """
    def __init__(self, hidden_size=128):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = self.flatten(x)  # Flatten the 28x28 image to a 784-dim vector
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MNISTFlow(FlowSpec):
    """
    A simplified Metaflow flow for training an MNIST digit classification model with PyTorch.
    """
    
    epochs = Parameter('epochs', default=3, help='Number of training epochs')
    batch_size = Parameter('batch_size', default=128, help='Batch size for training')
    learning_rate = Parameter('learning_rate', default=0.01, help='Learning rate')
    hidden_size = Parameter('hidden_size', default=128, help='Hidden layer size')
    
    @step
    def start(self):
        """
        Start the flow and print some information.
        """
        print("Starting simplified MNIST classification flow with PyTorch")
        print(f"Using PyTorch version: {torch.__version__}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Use CPU for simplicity
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """
        Load and prepare the MNIST dataset.
        """
        print("Loading MNIST dataset...")
        
        # Define transformations - just convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load training and test datasets
        self.train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        self.test_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
        
        self.next(self.build_model)
    
    @step
    def build_model(self):
        """
        Build a simple MLP model.
        """
        print("Building simple MLP model...")
        self.model = SimpleMLP(hidden_size=self.hidden_size).to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate,
            momentum=0.9
        )
        
        # Print model summary
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")
        
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """
        Train the MLP model on the MNIST dataset.
        """
        print(f"Training model for {self.epochs} epochs...")
        
        # Create a directory for this run's artifacts
        self.run_id = current.run_id
        self.artifact_dir = f"artifacts/{self.run_id}"
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        # Training loop
        self.train_losses = []
        self.train_accs = []
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} | Batch {batch_idx}/{len(self.train_loader)} | "
                          f"Loss: {running_loss/(batch_idx+1):.4f} | "
                          f"Acc: {100.*correct/total:.2f}%")
            
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100. * correct / total
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Save the model using pickle with .pkl extension
        self.model_path = f"{self.artifact_dir}/mnist_model.pkl"
        
        # Move model to CPU if it's not already there
        self.model = self.model.to('cpu')
        
        # Save the model using pickle
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {self.model_path}")
        
        self.next(self.evaluate_model)
    
    @step
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        """
        print("Evaluating model on test data...")
        
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        self.test_loss = test_loss / len(self.test_loader)
        self.test_accuracy = 100. * correct / total
        
        print(f"Test Loss: {self.test_loss:.4f} | Test Accuracy: {self.test_accuracy:.2f}%")
        
        # Generate classification report
        report = classification_report(all_targets, all_preds)
        print("\nClassification Report:")
        print(report)
        
        # Save classification report
        with open(f"{self.artifact_dir}/classification_report.txt", "w") as f:
            f.write(report)
        
        # Save some test data for visualization
        dataiter = iter(self.test_loader)
        self.test_images, self.test_labels = next(dataiter)
        self.test_images = self.test_images[:10].cpu()  # Get 10 images
        self.test_labels = self.test_labels[:10].cpu()
        
        # Get predictions for these images
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_images)
            _, self.test_preds = outputs.max(1)
        
        self.next(self.visualize_results)
    
    @step
    def visualize_results(self):
        """
        Visualize training history and some predictions.
        """
        print("Visualizing results...")
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.epochs + 1), self.train_accs, 'b-')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.epochs + 1), self.train_losses, 'r-')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.artifact_dir}/training_history.png")
        
        # Visualize sample predictions
        plt.figure(figsize=(12, 5))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            img = self.test_images[i].squeeze().numpy()
            plt.imshow(img, cmap='gray')
            
            pred_class = self.test_preds[i].item()
            true_class = self.test_labels[i].item()
            
            plt.title(f"Pred: {pred_class}\nTrue: {true_class}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.artifact_dir}/prediction_samples.png")
        
        # Create confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # Get predictions on the entire test set
            self.model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.numpy())
            
            # Create confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f"{self.artifact_dir}/confusion_matrix.png")
        except:
            print("Skipping confusion matrix visualization")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow and print final results.
        """
        print("\nSimplified MNIST Flow completed successfully!")
        print(f"Final test accuracy: {self.test_accuracy:.2f}%")
        print(f"All artifacts saved to: {self.artifact_dir}/")
        print(f"Model saved to: {self.model_path}")
        
        # Sample code to load the model
        print("\nTo load and use the model:")
        print("```python")
        print("import pickle")
        print("with open('path/to/mnist_model.pkl', 'rb') as f:")
        print("    model = pickle.load(f)")
        print("# Use the model for predictions")
        print("model.eval()")
        print("with torch.no_grad():")
        print("    predictions = model(your_data)")
        print("```")
        
        print("\nTo run this flow with custom parameters:")
        print("python mnist_flow.py run --epochs 5 --batch_size 64 --learning_rate 0.005 --hidden_size 256")

if __name__ == "__main__":
    MNISTFlow()