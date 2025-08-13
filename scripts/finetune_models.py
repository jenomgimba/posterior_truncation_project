# finetune_models.py - Create models with different overfitting levels
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import os
import json
import numpy as np
from tqdm import tqdm

class ModelFineTuner:
    """
    Fine-tune models to create different overfitting levels for membership inference research
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.training_stats = {}
        
    def load_datasets(self):
        """Load CIFAR datasets for fine-tuning"""
        print("=== Loading Datasets for Fine-tuning ===")
        
        # CIFAR-10 transforms - keep original size for faster training
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # CIFAR-100 transforms  
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        datasets_dict = {}
        
        try:
            # CIFAR-10
            cifar10_train = datasets.CIFAR10(root='./data', train=True, download=False, transform=cifar10_transform)
            cifar10_test = datasets.CIFAR10(root='./data', train=False, download=False, transform=cifar10_transform)
            
            # Use subset for faster training (you can increase this later)
            train_subset = Subset(cifar10_train, range(0, 10000))  # 10k samples
            test_subset = Subset(cifar10_test, range(0, 2000))     # 2k samples
            
            datasets_dict['cifar10'] = {
                'train': train_subset,
                'test': test_subset,
                'num_classes': 10
            }
            print(f"✓ CIFAR-10: {len(train_subset)} train, {len(test_subset)} test")
            
        except Exception as e:
            print(f"✗ CIFAR-10 failed: {e}")
        
        try:
            # CIFAR-100
            cifar100_train = datasets.CIFAR100(root='./data', train=True, download=False, transform=cifar100_transform)
            cifar100_test = datasets.CIFAR100(root='./data', train=False, download=False, transform=cifar100_transform)
            
            # Use subset for faster training
            train_subset = Subset(cifar100_train, range(0, 10000))  # 10k samples
            test_subset = Subset(cifar100_test, range(0, 2000))     # 2k samples
            
            datasets_dict['cifar100'] = {
                'train': train_subset,
                'test': test_subset,
                'num_classes': 100
            }
            print(f"✓ CIFAR-100: {len(train_subset)} train, {len(test_subset)} test")
            
        except Exception as e:
            print(f"✗ CIFAR-100 failed: {e}")
        
        return datasets_dict
    
    def create_base_model(self, architecture, num_classes):
        """Create base model for fine-tuning"""
        if architecture == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            # Adapt for CIFAR (smaller input, fewer classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        elif architecture == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1')
            # Adapt classifier for CIFAR
            model.classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return model
    
    def train_model(self, model, train_loader, test_loader, epochs=10, learning_rate=0.001, model_name="model"):
        """Train model and track overfitting"""
        print(f"\n=== Training {model_name} ===")
        
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_history = []
        test_history = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_acc = train_correct / train_total
            
            # Test phase
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_labels.size(0)
                    test_correct += (predicted == batch_labels).sum().item()
            
            test_acc = test_correct / test_total
            overfitting = train_acc - test_acc
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, Overfitting: {overfitting:.3f}")
            
            train_history.append(train_acc)
            test_history.append(test_acc)
        
        final_overfitting = train_history[-1] - test_history[-1]
        
        training_stats = {
            'final_train_acc': train_history[-1],
            'final_test_acc': test_history[-1],
            'overfitting_level': final_overfitting,
            'train_history': train_history,
            'test_history': test_history
        }
        
        return model, training_stats
    
    def create_research_models(self, datasets):
        """Create models with different overfitting levels"""
        print("=== Creating Research Models ===")
        
        # Define training configurations to create different overfitting levels
        configs = [
            {'epochs': 5, 'lr': 0.001, 'suffix': 'light'},      # Light training (less overfitting)
            {'epochs': 15, 'lr': 0.01, 'suffix': 'heavy'},     # Heavy training (more overfitting)
            {'epochs': 25, 'lr': 0.005, 'suffix': 'extreme'},  # Extreme training (most overfitting)
        ]
        
        architectures = ['resnet18', 'vgg16']
        
        for dataset_name, dataset_info in datasets.items():
            print(f"\n--- Processing {dataset_name.upper()} ---")
            
            train_loader = DataLoader(dataset_info['train'], batch_size=32, shuffle=True, num_workers=0)
            test_loader = DataLoader(dataset_info['test'], batch_size=32, shuffle=False, num_workers=0)
            
            for arch in architectures:
                for config in configs:
                    model_name = f"{arch}_{dataset_name}_{config['suffix']}"
                    
                    try:
                        # Create fresh model
                        model = self.create_base_model(arch, dataset_info['num_classes'])
                        
                        # Train model
                        trained_model, stats = self.train_model(
                            model, train_loader, test_loader,
                            epochs=config['epochs'],
                            learning_rate=config['lr'],
                            model_name=model_name
                        )
                        
                        # Store results
                        self.models[model_name] = trained_model
                        self.training_stats[model_name] = stats
                        
                        # Save model
                        os.makedirs('models/finetuned', exist_ok=True)
                        torch.save(trained_model.state_dict(), f'models/finetuned/{model_name}.pth')
                        
                        print(f"✓ {model_name}: Overfitting = {stats['overfitting_level']:.3f}")
                        
                    except Exception as e:
                        print(f"✗ {model_name} failed: {e}")
    
    def analyze_models(self):
        """Analyze overfitting levels of created models"""
        print("\n=== Model Analysis ===")
        
        analysis = {}
        
        for model_name, stats in self.training_stats.items():
            analysis[model_name] = {
                'train_accuracy': stats['final_train_acc'],
                'test_accuracy': stats['final_test_acc'],
                'overfitting_level': stats['overfitting_level'],
                'architecture': model_name.split('_')[0],
                'dataset': model_name.split('_')[1],
                'training_type': model_name.split('_')[2]
            }
        
        # Sort by overfitting level
        sorted_models = sorted(analysis.items(), key=lambda x: x[1]['overfitting_level'], reverse=True)
        
        print("Models ranked by overfitting level:")
        for name, info in sorted_models:
            print(f"{name:25s}: Overfitting = {info['overfitting_level']:.3f} "
                  f"(Train: {info['train_accuracy']:.3f}, Test: {info['test_accuracy']:.3f})")
        
        # Save analysis
        os.makedirs('results', exist_ok=True)
        with open('results/overfitting_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nAnalysis saved to results/overfitting_analysis.json")
        return analysis

def main():
    """Main function to create fine-tuned models"""
    print("Creating fine-tuned models for membership inference research...\n")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    finetuner = ModelFineTuner(device=device)
    
    # Load datasets
    datasets = finetuner.load_datasets()
    
    if not datasets:
        print("❌ No datasets available")
        return None
    
    # Create models with different overfitting levels
    finetuner.create_research_models(datasets)
    
    if finetuner.models:
        # Analyze results
        analysis = finetuner.analyze_models()
        
        print(f"\n🎉 Fine-tuning complete!")
        print(f"Created {len(finetuner.models)} models")
        print(f"Models saved in models/finetuned/")
        print("\nNext step: Implement membership inference attacks")
        
        return finetuner
    else:
        print("❌ Model fine-tuning failed")
        return None

if __name__ == "__main__":
    finetuner = main()