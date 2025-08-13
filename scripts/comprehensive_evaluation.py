# comprehensive_evaluation.py - Full evaluation with all truncation strategies
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for all models and truncation strategies
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.datasets = {}
        self.results = []
        
    def load_datasets(self):
        """Load datasets"""
        print("=== Loading Datasets ===")
        
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
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
    
    def load_model(self, model_path, num_classes):
        """Load ResNet model"""
        import torchvision.models as models
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def safe_softmax(self, logits):
        """Safe softmax computation"""
        logits = torch.clamp(logits, min=-50, max=50)
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - logits_max
        exp_logits = torch.exp(logits)
        softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        softmax_probs = torch.where(torch.isnan(softmax_probs) | torch.isinf(softmax_probs), 
                                   torch.ones_like(softmax_probs) / softmax_probs.size(1), 
                                   softmax_probs)
        return softmax_probs
    
    def get_model_outputs(self, model, data_loader, max_samples=500):
        """Get model outputs"""
        posteriors = []
        correctness = []
        
        model.eval()
        count = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in data_loader:
                if count >= max_samples:
                    break
                    
                batch_data = batch_data.to(self.device)
                outputs = model(batch_data)
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                batch_posteriors = self.safe_softmax(outputs)
                
                if torch.isnan(batch_posteriors).any() or torch.isinf(batch_posteriors).any():
                    continue
                
                batch_predictions = torch.argmax(batch_posteriors, dim=1)
                batch_correctness = (batch_predictions == batch_targets).float()
                
                posteriors.append(batch_posteriors.cpu())
                correctness.append(batch_correctness.cpu())
                count += len(batch_data)
        
        return {
            'posteriors': torch.cat(posteriors)[:max_samples],
            'correctness': torch.cat(correctness)[:max_samples]
        }
    
    def clean_data(self, data):
        """Clean NaN/Inf values"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        return np.where(np.isnan(data) | np.isinf(data), 0.5, data)
    
    def run_attacks(self, member_outputs, non_member_outputs):
        """Run all three attacks"""
        results = {}
        
        # PCA
        member_scores = self.clean_data(member_outputs['correctness'])
        non_member_scores = self.clean_data(non_member_outputs['correctness'])
        all_scores = np.concatenate([member_scores, non_member_scores])
        true_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
        
        if len(np.unique(all_scores)) > 1:
            results['PCA'] = roc_auc_score(true_labels, all_scores)
        else:
            results['PCA'] = 0.5
        
        # MPE
        member_posteriors = member_outputs['posteriors']
        non_member_posteriors = non_member_outputs['posteriors']
        eps = 1e-8
        
        member_entropy = -torch.sum(member_posteriors * torch.log(member_posteriors + eps), dim=1)
        non_member_entropy = -torch.sum(non_member_posteriors * torch.log(non_member_posteriors + eps), dim=1)
        
        member_entropy = self.clean_data(member_entropy)
        non_member_entropy = self.clean_data(non_member_entropy)
        
        max_entropy = np.log(member_posteriors.shape[1])
        member_scores = 1.0 - (member_entropy / max_entropy)
        non_member_scores = 1.0 - (non_member_entropy / max_entropy)
        
        all_scores = np.concatenate([member_scores, non_member_scores])
        
        if len(np.unique(all_scores)) > 1:
            results['MPE'] = roc_auc_score(true_labels, all_scores)
        else:
            results['MPE'] = 0.5
        
        # MLP
        try:
            member_features = self.clean_data(member_posteriors)
            non_member_features = self.clean_data(non_member_posteriors)
            
            X = np.vstack([member_features, non_member_features])
            y = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42)
            mlp.fit(X_train, y_train)
            y_pred_proba = mlp.predict_proba(X_test)[:, 1]
            
            if len(np.unique(y_test)) > 1:
                results['MLP'] = roc_auc_score(y_test, y_pred_proba)
            else:
                results['MLP'] = 0.5
                
        except:
            results['MLP'] = 0.5
        
        return results
    
    def apply_truncation(self, posteriors, strategy, **kwargs):
        """Apply truncation strategy"""
        posteriors = torch.clamp(posteriors, min=1e-8, max=1.0)
        
        if strategy == 'none':
            return posteriors
        elif strategy == 'top_k':
            k = kwargs.get('k', 3)
            k = min(k, posteriors.shape[1])
            top_k_values, top_k_indices = torch.topk(posteriors, k, dim=1)
            truncated = torch.zeros_like(posteriors)
            truncated.scatter_(1, top_k_indices, top_k_values)
            sum_vals = truncated.sum(dim=1, keepdim=True)
            sum_vals = torch.where(sum_vals == 0, torch.ones_like(sum_vals), sum_vals)
            return truncated / sum_vals
        elif strategy == 'confidence':
            threshold = kwargs.get('threshold', 0.1)
            truncated = posteriors.clone()
            truncated[truncated < threshold] = 0
            sum_vals = truncated.sum(dim=1, keepdim=True)
            sum_vals = torch.where(sum_vals == 0, torch.ones_like(sum_vals), sum_vals)
            return truncated / sum_vals
        
        return posteriors
    
    def evaluate_model(self, model_name, model_path, dataset_name):
        """Evaluate single model with all truncation strategies"""
        print(f"\n=== Evaluating {model_name} ===")
        
        try:
            # Load model and data
            dataset_info = self.datasets[dataset_name]
            model = self.load_model(model_path, dataset_info['num_classes'])
            
            member_dataset = Subset(dataset_info['train'], range(500))
            non_member_dataset = Subset(dataset_info['test'], range(500))
            
            member_loader = DataLoader(member_dataset, batch_size=16, shuffle=False, num_workers=0)
            non_member_loader = DataLoader(non_member_dataset, batch_size=16, shuffle=False, num_workers=0)
            
            member_outputs = self.get_model_outputs(model, member_loader)
            non_member_outputs = self.get_model_outputs(model, non_member_loader)
            
            # Get overfitting level
            with open('results/overfitting_analysis.json', 'r') as f:
                overfitting_data = json.load(f)
            overfitting_level = overfitting_data.get(model_name, {}).get('overfitting_level', 0)
            
            # Test all truncation strategies
            truncation_configs = [
                {'strategy': 'none'},
                {'strategy': 'top_k', 'k': 1},
                {'strategy': 'top_k', 'k': 3},
                {'strategy': 'top_k', 'k': 5},
                {'strategy': 'confidence', 'threshold': 0.1},
                {'strategy': 'confidence', 'threshold': 0.3},
                {'strategy': 'confidence', 'threshold': 0.5}
            ]
            
            for config in truncation_configs:
                # Apply truncation
                if config['strategy'] == 'none':
                    trunc_member = member_outputs['posteriors']
                    trunc_non_member = non_member_outputs['posteriors']
                    config_name = 'baseline'
                else:
                    trunc_member = self.apply_truncation(member_outputs['posteriors'], **config)
                    trunc_non_member = self.apply_truncation(non_member_outputs['posteriors'], **config)
                    config_name = f"{config['strategy']}"
                    if 'k' in config:
                        config_name += f"_k{config['k']}"
                    elif 'threshold' in config:
                        config_name += f"_t{config['threshold']}"
                
                # Create truncated outputs
                trunc_member_outputs = {'posteriors': trunc_member, 'correctness': member_outputs['correctness']}
                trunc_non_member_outputs = {'posteriors': trunc_non_member, 'correctness': non_member_outputs['correctness']}
                
                # Run attacks
                attack_results = self.run_attacks(trunc_member_outputs, trunc_non_member_outputs)
                
                # Store results
                for attack_method, auc in attack_results.items():
                    self.results.append({
                        'model_name': model_name,
                        'dataset': dataset_name,
                        'overfitting_level': overfitting_level,
                        'truncation_strategy': config_name,
                        'attack_method': attack_method,
                        'auc': auc
                    })
                
                print(f"  {config_name:15s}: PCA={attack_results['PCA']:.3f}, MPE={attack_results['MPE']:.3f}, MLP={attack_results['MLP']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False
    
    def run_comprehensive_evaluation(self):
        """Run evaluation on all working models"""
        print("=== Comprehensive Evaluation ===")
        
        # Models to evaluate
        models_to_test = [
            ('resnet18_cifar10_light', 'cifar10'),
            ('resnet18_cifar10_heavy', 'cifar10'),
            ('resnet18_cifar10_extreme', 'cifar10'),
            ('resnet18_cifar100_light', 'cifar100'),
        ]
        
        for model_name, dataset_name in models_to_test:
            model_path = f'models/finetuned/{model_name}.pth'
            if os.path.exists(model_path):
                self.evaluate_model(model_name, model_path, dataset_name)
        
        # Save results
        df = pd.DataFrame(self.results)
        os.makedirs('results', exist_ok=True)
        df.to_csv('results/comprehensive_results.csv', index=False)
        
        with open('results/comprehensive_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n🎉 Comprehensive evaluation complete!")
        print(f"Results saved to results/comprehensive_results.csv")
        print(f"Total experiments: {len(self.results)}")
        
        return df
    
    def create_visualizations(self, df):
        """Create all visualizations for your paper"""
        print("\n=== Creating Visualizations ===")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create results directory
        os.makedirs('results/figures', exist_ok=True)
        
        # 1. Overfitting vs Attack Success
        baseline_results = df[df['truncation_strategy'] == 'baseline']
        
        plt.figure(figsize=(12, 4))
        
        for i, attack in enumerate(['PCA', 'MPE', 'MLP']):
            plt.subplot(1, 3, i+1)
            attack_data = baseline_results[baseline_results['attack_method'] == attack]
            
            scatter = plt.scatter(attack_data['overfitting_level'], attack_data['auc'], 
                                s=100, alpha=0.7, label=attack)
            
            # Add trend line
            z = np.polyfit(attack_data['overfitting_level'], attack_data['auc'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(attack_data['overfitting_level'].min(), 
                                attack_data['overfitting_level'].max(), 100)
            plt.plot(x_trend, p(x_trend), "--", alpha=0.7)
            
            plt.xlabel('Overfitting Level')
            plt.ylabel('AUC')
            plt.title(f'{attack} Attack')
            plt.grid(True, alpha=0.3)
            
            # Add model labels
            for _, row in attack_data.iterrows():
                plt.annotate(row['model_name'].replace('resnet18_', ''), 
                           (row['overfitting_level'], row['auc']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/figures/overfitting_vs_attack_success.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Truncation Effectiveness Heatmap
        plt.figure(figsize=(14, 8))
        
        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(values='auc', 
                                    index=['model_name', 'attack_method'], 
                                    columns='truncation_strategy', 
                                    aggfunc='mean')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=0.5, vmin=0.4, vmax=0.8)
        plt.title('Attack Success (AUC) Across Truncation Strategies')
        plt.xlabel('Truncation Strategy')
        plt.ylabel('Model and Attack Method')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/figures/truncation_effectiveness_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Defense Effectiveness by Model
        plt.figure(figsize=(15, 10))
        
        models = df['model_name'].unique()
        
        for i, model in enumerate(models):
            plt.subplot(2, 2, i+1)
            model_data = df[df['model_name'] == model]
            
            # Group by truncation strategy and attack method
            strategy_order = ['baseline', 'top_k_k1', 'top_k_k3', 'top_k_k5', 
                            'confidence_t0.1', 'confidence_t0.3', 'confidence_t0.5']
            
            for attack in ['PCA', 'MPE', 'MLP']:
                attack_data = model_data[model_data['attack_method'] == attack]
                aucs = []
                strategies = []
                
                for strategy in strategy_order:
                    strategy_auc = attack_data[attack_data['truncation_strategy'] == strategy]['auc']
                    if len(strategy_auc) > 0:
                        aucs.append(strategy_auc.iloc[0])
                        strategies.append(strategy)
                
                plt.plot(range(len(aucs)), aucs, marker='o', label=attack, linewidth=2)
            
            plt.title(f'{model.replace("resnet18_", "")}')
            plt.xlabel('Truncation Strategy')
            plt.ylabel('AUC')
            plt.xticks(range(len(strategies)), [s.replace('_', '\n') for s in strategies], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0.4, 0.8)
        
        plt.tight_layout()
        plt.savefig('results/figures/defense_effectiveness_by_model.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Summary Statistics Table
        summary_stats = []
        
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            baseline_data = model_data[model_data['truncation_strategy'] == 'baseline']
            
            overfitting = model_data['overfitting_level'].iloc[0]
            baseline_avg_auc = baseline_data['auc'].mean()
            
            # Best defense (lowest average AUC across attacks)
            defense_data = model_data[model_data['truncation_strategy'] != 'baseline']
            best_defense_auc = defense_data.groupby('truncation_strategy')['auc'].mean().min()
            best_defense_name = defense_data.groupby('truncation_strategy')['auc'].mean().idxmin()
            
            defense_improvement = baseline_avg_auc - best_defense_auc
            
            summary_stats.append({
                'Model': model.replace('resnet18_', ''),
                'Overfitting': f"{overfitting:.3f}",
                'Baseline AUC': f"{baseline_avg_auc:.3f}",
                'Best Defense': best_defense_name,
                'Best Defense AUC': f"{best_defense_auc:.3f}",
                'Improvement': f"{defense_improvement:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv('results/summary_statistics.csv', index=False)
        
        print("✓ Overfitting vs Attack Success plot")
        print("✓ Truncation Effectiveness Heatmap") 
        print("✓ Defense Effectiveness by Model")
        print("✓ Summary Statistics Table")
        print("\nAll figures saved in results/figures/")
        
        return summary_df

def main():
    """Main comprehensive evaluation"""
    evaluator = ComprehensiveEvaluator()
    evaluator.load_datasets()
    
    # Run comprehensive evaluation
    df = evaluator.run_comprehensive_evaluation()
    
    if len(df) > 0:
        # Create visualizations
        summary = evaluator.create_visualizations(df)
        
        print("\n=== Summary Table ===")
        print(summary.to_string(index=False))
        
        return evaluator, df, summary
    else:
        print("❌ No results to visualize")
        return None, None, None

if __name__ == "__main__":
    evaluator, df, summary = main()