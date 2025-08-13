# create_research_models.py - Create models for your membership inference research
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import os
import json

class ResearchModelCreator:
    """
    Create and manage models for membership inference research
    """
    
    def __init__(self):
        self.models = {}
        self.datasets = {}
        
    def create_cifar10_models(self):
        """Create models adapted for CIFAR-10 (10 classes)"""
        print("=== Creating CIFAR-10 Models ===")
        
        # ResNet models for CIFAR-10
        resnet18_cifar10 = models.resnet18(weights='IMAGENET1K_V1')
        resnet18_cifar10.fc = nn.Linear(resnet18_cifar10.fc.in_features, 10)
        
        resnet50_cifar10 = models.resnet50(weights='IMAGENET1K_V1') 
        resnet50_cifar10.fc = nn.Linear(resnet50_cifar10.fc.in_features, 10)
        
        # VGG models for CIFAR-10
        vgg16_cifar10 = models.vgg16(weights='IMAGENET1K_V1')
        vgg16_cifar10.classifier[-1] = nn.Linear(vgg16_cifar10.classifier[-1].in_features, 10)
        
        vgg19_cifar10 = models.vgg19(weights='IMAGENET1K_V1')
        vgg19_cifar10.classifier[-1] = nn.Linear(vgg19_cifar10.classifier[-1].in_features, 10)
        
        cifar10_models = {
            'resnet18_cifar10': resnet18_cifar10,
            'resnet50_cifar10': resnet50_cifar10,
            'vgg16_cifar10': vgg16_cifar10,
            'vgg19_cifar10': vgg19_cifar10
        }
        
        # Test each model
        dummy_input = torch.randn(1, 3, 224, 224)  # ImageNet size input
        
        for name, model in cifar10_models.items():
            try:
                model.eval()
                with torch.no_grad():
                    output = model(dummy_input)
                params = sum(p.numel() for p in model.parameters())
                print(f"✓ {name}: {params:,} parameters, output shape: {output.shape}")
                self.models[name] = model
            except Exception as e:
                print(f"✗ {name} failed: {e}")
    
    def create_cifar100_models(self):
        """Create models adapted for CIFAR-100 (100 classes)"""
        print("\n=== Creating CIFAR-100 Models ===")
        
        # ResNet models for CIFAR-100
        resnet18_cifar100 = models.resnet18(weights='IMAGENET1K_V1')
        resnet18_cifar100.fc = nn.Linear(resnet18_cifar100.fc.in_features, 100)
        
        resnet50_cifar100 = models.resnet50(weights='IMAGENET1K_V1')
        resnet50_cifar100.fc = nn.Linear(resnet50_cifar100.fc.in_features, 100)
        
        # VGG models for CIFAR-100
        vgg16_cifar100 = models.vgg16(weights='IMAGENET1K_V1')
        vgg16_cifar100.classifier[-1] = nn.Linear(vgg16_cifar100.classifier[-1].in_features, 100)
        
        cifar100_models = {
            'resnet18_cifar100': resnet18_cifar100,
            'resnet50_cifar100': resnet50_cifar100,
            'vgg16_cifar100': vgg16_cifar100
        }
        
        # Test each model
        dummy_input = torch.randn(1, 3, 224, 224)
        
        for name, model in cifar100_models.items():
            try:
                model.eval()
                with torch.no_grad():
                    output = model(dummy_input)
                params = sum(p.numel() for p in model.parameters())
                print(f"✓ {name}: {params:,} parameters, output shape: {output.shape}")
                self.models[name] = model
            except Exception as e:
                print(f"✗ {name} failed: {e}")
    
    def load_datasets(self):
        """Load and prepare datasets for membership inference"""
        print("\n=== Loading Datasets ===")
        
        # CIFAR-10 transforms (resize to 224x224 for ImageNet models)
        cifar10_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # CIFAR-100 transforms
        cifar100_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        try:
            # Load CIFAR-10
            cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar10_transform)
            cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar10_transform)
            
            self.datasets['cifar10'] = {
                'train': cifar10_train,  # Members
                'test': cifar10_test,    # Non-members
                'num_classes': 10
            }
            print(f"✓ CIFAR-10: {len(cifar10_train)} train, {len(cifar10_test)} test")
            
        except Exception as e:
            print(f"✗ CIFAR-10 loading failed: {e}")
        
        try:
            # Load CIFAR-100
            cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar100_transform)
            cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar100_transform)
            
            self.datasets['cifar100'] = {
                'train': cifar100_train,  # Members
                'test': cifar100_test,    # Non-members  
                'num_classes': 100
            }
            print(f"✓ CIFAR-100: {len(cifar100_train)} train, {len(cifar100_test)} test")
            
        except Exception as e:
            print(f"✗ CIFAR-100 loading failed: {e}")
    
    def test_model_on_data(self, model_name, dataset_name, num_samples=100):
        """Test a specific model on a specific dataset"""
        if model_name not in self.models:
            print(f"Model {model_name} not available")
            return None
        
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not available")
            return None
        
        model = self.models[model_name]
        dataset = self.datasets[dataset_name]['test']  # Use test set for quick test
        
        # Create small data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=0
        )
        
        model.eval()
        correct = 0
        total = 0
        all_posteriors = []
        
        with torch.no_grad():
            for i, (data, targets) in enumerate(data_loader):
                if total >= num_samples:
                    break
                
                outputs = model(data)
                posteriors = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(posteriors, dim=1)
                
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                all_posteriors.extend(posteriors.max(dim=1)[0].tolist())
        
        accuracy = correct / total
        avg_confidence = sum(all_posteriors) / len(all_posteriors)
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'samples_tested': total
        }
    
    def create_model_summary(self):
        """Create summary of all available models and their performance"""
        print("\n=== Model Performance Summary ===")
        
        summary = {}
        
        for model_name in self.models.keys():
            summary[model_name] = {}
            
            # Determine which dataset this model is for
            if 'cifar10' in model_name:
                dataset_name = 'cifar10'
            elif 'cifar100' in model_name:
                dataset_name = 'cifar100'
            else:
                continue
            
            # Test model performance
            performance = self.test_model_on_data(model_name, dataset_name)
            if performance:
                summary[model_name] = performance
                print(f"{model_name:20s}: Acc={performance['accuracy']:.3f}, Conf={performance['avg_confidence']:.3f}")
        
        # Save summary
        os.makedirs('results', exist_ok=True)
        with open('results/model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to results/model_summary.json")
        return summary
    
    def save_models(self):
        """Save models for later use"""
        print("\n=== Saving Models ===")
        
        os.makedirs('models/research_models', exist_ok=True)
        
        for name, model in self.models.items():
            try:
                torch.save(model.state_dict(), f'models/research_models/{name}.pth')
                print(f"✓ Saved {name}")
            except Exception as e:
                print(f"✗ Failed to save {name}: {e}")

def main():
    """Main function to create research setup"""
    print("Creating models for membership inference research...\n")
    
    creator = ResearchModelCreator()
    
    # Create models
    creator.create_cifar10_models()
    creator.create_cifar100_models()
    
    # Load datasets
    creator.load_datasets()
    
    # Test and summarize
    if creator.models and creator.datasets:
        summary = creator.create_model_summary()
        creator.save_models()
        
        print("\n🎉 Research models created successfully!")
        print(f"Available models: {len(creator.models)}")
        print(f"Available datasets: {len(creator.datasets)}")
        print("\nNext step: Implement membership inference attacks")
        
        return creator
    else:
        print("❌ Model creation failed")
        return None

if __name__ == "__main__":
    creator = main()