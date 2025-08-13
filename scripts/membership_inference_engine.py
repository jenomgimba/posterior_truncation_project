# fixed_membership_inference_engine.py - NaN-safe version
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets
import warnings
warnings.filterwarnings('ignore')

class RobustMembershipInferenceEngine:
    """
    Robust membership inference engine that handles NaN values and numerical instabilities
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.datasets = {}
        self.results = {}
        
    def load_model(self, model_path, architecture, num_classes):
        """Load a trained model"""
        if architecture == 'resnet18':
            import torchvision.models as models
            model = models.resnet18(weights=None)
            # Adapt for CIFAR
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Architecture {architecture} not supported")
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def load_datasets(self):
        """Load datasets for membership inference"""
        print("=== Loading Datasets ===")
        
        # CIFAR-10
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # CIFAR-100
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        self.datasets = {
            'cifar10': {
                'train': datasets.CIFAR10(root='./data', train=True, download=False, transform=cifar10_transform),
                'test': datasets.CIFAR10(root='./data', train=False, download=False, transform=cifar10_transform),
                'num_classes': 10
            },
            'cifar100': {
                'train': datasets.CIFAR100(root='./data', train=True, download=False, transform=cifar100_transform),
                'test': datasets.CIFAR100(root='./data', train=False, download=False, transform=cifar100_transform),
                'num_classes': 100
            }
        }
        print("✓ Datasets loaded")
    
    def safe_softmax(self, logits, temperature=1.0):
        """Safe softmax that handles extreme values"""
        # Clip extreme values to prevent overflow/underflow
        logits = torch.clamp(logits, min=-50, max=50)
        logits = logits / temperature
        
        # Subtract max for numerical stability
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - logits_max
        
        # Compute softmax
        exp_logits = torch.exp(logits)
        softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        
        # Handle any remaining NaN/Inf values
        softmax_probs = torch.where(torch.isnan(softmax_probs) | torch.isinf(softmax_probs), 
                                   torch.ones_like(softmax_probs) / softmax_probs.size(1), 
                                   softmax_probs)
        
        return softmax_probs
    
    def get_model_outputs(self, model, data_loader, max_samples=1000):
        """Get model outputs for membership inference with NaN handling"""
        posteriors = []
        predictions = []
        targets = []
        correctness = []
        
        model.eval()
        count = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in data_loader:
                if count >= max_samples:
                    break
                    
                batch_data = batch_data.to(self.device)
                
                try:
                    outputs = model(batch_data)
                    
                    # Check for NaN/Inf in raw outputs
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"Warning: NaN/Inf detected in model outputs, skipping batch")
                        continue
                    
                    # Use safe softmax
                    batch_posteriors = self.safe_softmax(outputs)
                    
                    # Final check for NaN/Inf
                    if torch.isnan(batch_posteriors).any() or torch.isinf(batch_posteriors).any():
                        print(f"Warning: NaN/Inf detected in posteriors, skipping batch")
                        continue
                    
                    batch_predictions = torch.argmax(batch_posteriors, dim=1)
                    batch_correctness = (batch_predictions == batch_targets).float()
                    
                    posteriors.append(batch_posteriors.cpu())
                    predictions.append(batch_predictions.cpu())
                    targets.append(batch_targets)
                    correctness.append(batch_correctness.cpu())
                    
                    count += len(batch_data)
                    
                except Exception as e:
                    print(f"Warning: Error processing batch: {e}")
                    continue
        
        if len(posteriors) == 0:
            raise ValueError("No valid batches processed - all contained NaN/Inf values")
        
        return {
            'posteriors': torch.cat(posteriors)[:max_samples],
            'predictions': torch.cat(predictions)[:max_samples],
            'targets': torch.cat(targets)[:max_samples],
            'correctness': torch.cat(correctness)[:max_samples]
        }
    
    def clean_data(self, data):
        """Remove NaN and Inf values from data"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        # Replace NaN and Inf with finite values
        data = np.where(np.isnan(data) | np.isinf(data), 0.5, data)
        
        return data
    
    def prediction_correctness_attack(self, member_outputs, non_member_outputs):
        """Prediction Correctness Attack (PCA) with NaN handling"""
        member_scores = self.clean_data(member_outputs['correctness'])
        non_member_scores = self.clean_data(non_member_outputs['correctness'])
        
        # Combine scores (members=1, non-members=0)
        all_scores = np.concatenate([member_scores, non_member_scores])
        true_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
        
        # Calculate AUC only if we have variation in scores
        if len(np.unique(all_scores)) > 1:
            auc = roc_auc_score(true_labels, all_scores)
        else:
            auc = 0.5  # Random performance if no variation
        
        return {
            'attack_method': 'PCA',
            'auc': auc,
            'member_scores': member_scores,
            'non_member_scores': non_member_scores
        }
    
    def modified_prediction_entropy_attack(self, member_outputs, non_member_outputs):
        """Modified Prediction Entropy Attack (MPE) with NaN handling"""
        member_posteriors = member_outputs['posteriors']
        non_member_posteriors = non_member_outputs['posteriors']
        
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        member_posteriors = torch.clamp(member_posteriors, min=eps, max=1.0)
        non_member_posteriors = torch.clamp(non_member_posteriors, min=eps, max=1.0)
        
        # Calculate entropy
        member_entropy = -torch.sum(member_posteriors * torch.log(member_posteriors + eps), dim=1)
        non_member_entropy = -torch.sum(non_member_posteriors * torch.log(non_member_posteriors + eps), dim=1)
        
        # Clean data
        member_entropy = self.clean_data(member_entropy)
        non_member_entropy = self.clean_data(non_member_entropy)
        
        # Convert entropy to membership scores (lower entropy = higher membership probability)
        max_entropy = np.log(member_posteriors.shape[1])
        member_scores = 1.0 - (member_entropy / max_entropy)
        non_member_scores = 1.0 - (non_member_entropy / max_entropy)
        
        # Clean scores
        member_scores = self.clean_data(member_scores)
        non_member_scores = self.clean_data(non_member_scores)
        
        # Combine for AUC calculation
        all_scores = np.concatenate([member_scores, non_member_scores])
        true_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
        
        if len(np.unique(all_scores)) > 1:
            auc = roc_auc_score(true_labels, all_scores)
        else:
            auc = 0.5
        
        return {
            'attack_method': 'MPE',
            'auc': auc,
            'member_scores': member_scores,
            'non_member_scores': non_member_scores
        }
    
    def mlp_attack(self, member_outputs, non_member_outputs):
        """MLP-based Attack with robust error handling"""
        # Prepare training data
        member_features = member_outputs['posteriors'].numpy()
        non_member_features = non_member_outputs['posteriors'].numpy()
        
        # Clean features
        member_features = self.clean_data(member_features)
        non_member_features = self.clean_data(non_member_features)
        
        X = np.vstack([member_features, non_member_features])
        y = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
        
        # Check for valid data
        if np.all(np.isnan(X)) or np.all(np.isinf(X)):
            return {
                'attack_method': 'MLP',
                'auc': 0.5,
                'member_scores': np.full(len(member_features), 0.5),
                'non_member_scores': np.full(len(non_member_features), 0.5),
                'test_auc': 0.5,
                'model': None
            }
        
        # Split for training and testing
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        except ValueError:
            # If stratify fails, try without stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train MLP with more robust settings
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),  # Smaller network
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,  # Fewer iterations
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        try:
            mlp.fit(X_train, y_train)
            
            # Get membership probabilities
            y_pred_proba = mlp.predict_proba(X_test)[:, 1]
            
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = 0.5
            
            # Get scores for original member/non-member sets
            member_scores = mlp.predict_proba(member_features)[:, 1]
            non_member_scores = mlp.predict_proba(non_member_features)[:, 1]
            
            return {
                'attack_method': 'MLP',
                'auc': auc,
                'member_scores': member_scores,
                'non_member_scores': non_member_scores,
                'test_auc': auc,
                'model': mlp
            }
            
        except Exception as e:
            print(f"MLP training failed: {e}")
            return {
                'attack_method': 'MLP',
                'auc': 0.5,
                'member_scores': np.full(len(member_features), 0.5),
                'non_member_scores': np.full(len(non_member_features), 0.5),
                'test_auc': 0.5,
                'model': None
            }
    
    def apply_posterior_truncation(self, posteriors, strategy, **kwargs):
        """Apply posterior truncation defense with NaN handling"""
        posteriors = torch.clamp(posteriors, min=1e-8, max=1.0)  # Prevent extreme values
        
        if strategy == 'none':
            return posteriors
        
        elif strategy == 'top_k':
            k = kwargs.get('k', 3)
            k = min(k, posteriors.shape[1])  # Ensure k doesn't exceed number of classes
            
            # Get top-k values and indices
            top_k_values, top_k_indices = torch.topk(posteriors, k, dim=1)
            # Create truncated posteriors
            truncated = torch.zeros_like(posteriors)
            truncated.scatter_(1, top_k_indices, top_k_values)
            # Renormalize
            sum_vals = truncated.sum(dim=1, keepdim=True)
            sum_vals = torch.where(sum_vals == 0, torch.ones_like(sum_vals), sum_vals)
            truncated = truncated / sum_vals
            return truncated
        
        elif strategy == 'confidence':
            threshold = kwargs.get('threshold', 0.1)
            # Set values below threshold to zero
            truncated = posteriors.clone()
            truncated[truncated < threshold] = 0
            # Renormalize
            sum_vals = truncated.sum(dim=1, keepdim=True)
            sum_vals = torch.where(sum_vals == 0, torch.ones_like(sum_vals), sum_vals)
            truncated = truncated / sum_vals
            return truncated
        
        else:
            return posteriors  # Fallback to no truncation
    
    def evaluate_single_model(self, model_name, model_path, dataset_name, sample_size=500):
        """Evaluate a single model - simplified version"""
        print(f"\n=== Evaluating {model_name} on {dataset_name} ===")
        
        try:
            # Load model
            dataset_info = self.datasets[dataset_name]
            model = self.load_model(model_path, 'resnet18', dataset_info['num_classes'])
            
            # Create smaller data loaders for testing
            member_dataset = Subset(dataset_info['train'], range(sample_size))
            non_member_dataset = Subset(dataset_info['test'], range(sample_size))
            
            member_loader = DataLoader(member_dataset, batch_size=16, shuffle=False, num_workers=0)
            non_member_loader = DataLoader(non_member_dataset, batch_size=16, shuffle=False, num_workers=0)
            
            # Get model outputs
            member_outputs = self.get_model_outputs(model, member_loader, sample_size)
            non_member_outputs = self.get_model_outputs(model, non_member_loader, sample_size)
            
            print(f"✓ Got outputs: {len(member_outputs['posteriors'])} members, {len(non_member_outputs['posteriors'])} non-members")
            
            # Run baseline attacks
            results = {
                'model_name': model_name,
                'dataset': dataset_name,
                'baseline_attacks': {}
            }
            
            print("Running attacks...")
            results['baseline_attacks']['PCA'] = self.prediction_correctness_attack(member_outputs, non_member_outputs)
            results['baseline_attacks']['MPE'] = self.modified_prediction_entropy_attack(member_outputs, non_member_outputs)
            results['baseline_attacks']['MLP'] = self.mlp_attack(member_outputs, non_member_outputs)
            
            # Test one truncation strategy
            print("Testing top-3 truncation...")
            truncated_member_posteriors = self.apply_posterior_truncation(member_outputs['posteriors'], 'top_k', k=3)
            truncated_non_member_posteriors = self.apply_posterior_truncation(non_member_outputs['posteriors'], 'top_k', k=3)
            
            truncated_member_outputs = member_outputs.copy()
            truncated_member_outputs['posteriors'] = truncated_member_posteriors
            
            truncated_non_member_outputs = non_member_outputs.copy()
            truncated_non_member_outputs['posteriors'] = truncated_non_member_posteriors
            
            results['truncated_attacks'] = {
                'top_k_3': {
                    'PCA': self.prediction_correctness_attack(truncated_member_outputs, truncated_non_member_outputs),
                    'MPE': self.modified_prediction_entropy_attack(truncated_member_outputs, truncated_non_member_outputs),
                    'MLP': self.mlp_attack(truncated_member_outputs, truncated_non_member_outputs)
                }
            }
            
            # Print results
            print("Baseline Results:")
            for attack, result in results['baseline_attacks'].items():
                print(f"  {attack}: AUC = {result['auc']:.3f}")
            
            print("Truncated Results (top-3):")
            for attack, result in results['truncated_attacks']['top_k_3'].items():
                print(f"  {attack}: AUC = {result['auc']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            return None
    
    def run_evaluation(self):
        """Run simplified evaluation"""
        print("=== Robust Membership Inference Evaluation ===")
        
        # Test just the best models first
        test_models = [
            ('resnet18_cifar10_light', 'cifar10'),  # Start with least overfitted
            ('resnet18_cifar10_extreme', 'cifar10'),
            ('resnet18_cifar100_light', 'cifar100')
        ]
        
        all_results = []
        
        for model_name, dataset_name in test_models:
            model_path = f'models/finetuned/{model_name}.pth'
            
            if os.path.exists(model_path):
                result = self.evaluate_single_model(model_name, model_path, dataset_name)
                if result:
                    all_results.append(result)
            else:
                print(f"✗ Model file not found: {model_path}")
        
        # Save results
        if all_results:
            os.makedirs('results', exist_ok=True)
            with open('results/mia_robust_results.json', 'w') as f:
                serializable_results = []
                for result in all_results:
                    serializable_result = self._make_serializable(result)
                    serializable_results.append(serializable_result)
                json.dump(serializable_results, f, indent=2)
            
            print(f"\n🎉 Evaluation complete!")
            print(f"Results saved to results/mia_robust_results.json")
            print(f"Successfully evaluated {len(all_results)} models")
        
        return all_results
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() if k != 'model'}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

def main():
    """Main function"""
    print("Starting Robust Membership Inference Engine...\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    engine = RobustMembershipInferenceEngine(device=device)
    engine.load_datasets()
    
    results = engine.run_evaluation()
    
    if results:
        print("\n=== Summary ===")
        for result in results:
            model_name = result['model_name']
            baseline_aucs = [attack['auc'] for attack in result['baseline_attacks'].values()]
            avg_baseline_auc = np.mean(baseline_aucs)
            print(f"{model_name:25s}: Avg AUC = {avg_baseline_auc:.3f}")
        
        return engine, results
    else:
        print("❌ No results generated")
        return None, None

if __name__ == "__main__":
    engine, results = main()