# performance_analysis.py - Latency and throughput analysis
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computational_performance_analysis():
    """
    Analyze computational overhead of truncation strategies
    """
    print("=== Computational Performance Analysis ===")
    
    # Simulate different truncation operations
    def time_truncation_operation(posteriors, strategy, **kwargs):
        """Time a specific truncation operation"""
        times = []
        
        for _ in range(100):  # Multiple runs for averaging
            start_time = time.time()
            
            if strategy == 'none':
                result = posteriors
            elif strategy == 'top_k':
                k = kwargs.get('k', 3)
                top_k_values, top_k_indices = torch.topk(posteriors, k, dim=1)
                result = torch.zeros_like(posteriors)
                result.scatter_(1, top_k_indices, top_k_values)
                result = result / result.sum(dim=1, keepdim=True)
            elif strategy == 'confidence':
                threshold = kwargs.get('threshold', 0.1)
                result = posteriors.clone()
                result[result < threshold] = 0
                result = result / result.sum(dim=1, keepdim=True)
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return np.mean(times), np.std(times)
    
    # Test different batch sizes and number of classes
    test_configs = [
        {'batch_size': 32, 'num_classes': 10, 'dataset': 'CIFAR-10'},
        {'batch_size': 32, 'num_classes': 100, 'dataset': 'CIFAR-100'},
        {'batch_size': 64, 'num_classes': 10, 'dataset': 'CIFAR-10'},
        {'batch_size': 128, 'num_classes': 10, 'dataset': 'CIFAR-10'},
    ]
    
    truncation_strategies = [
        {'strategy': 'none'},
        {'strategy': 'top_k', 'k': 1},
        {'strategy': 'top_k', 'k': 3},
        {'strategy': 'top_k', 'k': 5},
        {'strategy': 'confidence', 'threshold': 0.1},
        {'strategy': 'confidence', 'threshold': 0.3},
        {'strategy': 'confidence', 'threshold': 0.5}
    ]
    
    performance_results = []
    
    print("\nTiming truncation operations...")
    
    for config in test_configs:
        print(f"\n{config['dataset']} (batch={config['batch_size']}, classes={config['num_classes']}):")
        
        # Generate random posteriors
        posteriors = torch.softmax(torch.randn(config['batch_size'], config['num_classes']), dim=1)
        
        for strategy_config in truncation_strategies:
            strategy_name = strategy_config['strategy']
            if 'k' in strategy_config:
                strategy_name += f"_k{strategy_config['k']}"
            elif 'threshold' in strategy_config:
                strategy_name += f"_t{strategy_config['threshold']}"
            
            mean_time, std_time = time_truncation_operation(posteriors, **strategy_config)
            
            # Calculate throughput (samples per second)
            throughput = config['batch_size'] / (mean_time / 1000) if mean_time > 0 else 0
            
            performance_results.append({
                'dataset': config['dataset'],
                'batch_size': config['batch_size'],
                'num_classes': config['num_classes'],
                'strategy': strategy_name,
                'mean_latency_ms': mean_time,
                'std_latency_ms': std_time,
                'throughput_samples_per_sec': throughput
            })
            
            print(f"  {strategy_name:15s}: {mean_time:.3f}±{std_time:.3f} ms, {throughput:.0f} samples/sec")
    
    # Save performance results
    perf_df = pd.DataFrame(performance_results)
    perf_df.to_csv('results/performance_analysis.csv', index=False)
    
    # Create performance visualization
    plt.figure(figsize=(15, 10))
    
    # Latency comparison
    plt.subplot(2, 2, 1)
    baseline_latency = perf_df[perf_df['strategy'] == 'none']['mean_latency_ms'].iloc[0]
    
    strategy_latencies = perf_df[perf_df['batch_size'] == 32].groupby('strategy')['mean_latency_ms'].mean()
    overhead = (strategy_latencies - baseline_latency) / baseline_latency * 100
    
    overhead.plot(kind='bar')
    plt.title('Latency Overhead vs Baseline (%)')
    plt.ylabel('Overhead (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Throughput comparison
    plt.subplot(2, 2, 2)
    strategy_throughput = perf_df[perf_df['batch_size'] == 32].groupby('strategy')['throughput_samples_per_sec'].mean()
    strategy_throughput.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Throughput by Strategy')
    plt.ylabel('Samples/Second')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Batch size scaling
    plt.subplot(2, 2, 3)
    for strategy in ['none', 'top_k_k3', 'confidence_t0.3']:
        strategy_data = perf_df[perf_df['strategy'] == strategy]
        plt.plot(strategy_data['batch_size'], strategy_data['throughput_samples_per_sec'], 
                marker='o', label=strategy)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/sec)')
    plt.title('Scalability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Latency vs Defense Effectiveness
    plt.subplot(2, 2, 4)
    
    # Load attack results for comparison
    try:
        attack_df = pd.read_csv('results/comprehensive_results.csv')
        
        # Calculate average defense effectiveness for each strategy
        baseline_aucs = attack_df[attack_df['truncation_strategy'] == 'baseline']['auc']
        baseline_avg = baseline_aucs.mean()
        
        defense_effectiveness = []
        latencies = []
        strategy_names = []
        
        for strategy in perf_df['strategy'].unique():
            if strategy != 'none':
                # Get defense effectiveness
                strategy_aucs = attack_df[attack_df['truncation_strategy'] == strategy]['auc']
                if len(strategy_aucs) > 0:
                    defense_effect = baseline_avg - strategy_aucs.mean()
                    
                    # Get latency
                    strategy_latency = perf_df[perf_df['strategy'] == strategy]['mean_latency_ms'].mean()
                    
                    defense_effectiveness.append(defense_effect)
                    latencies.append(strategy_latency)
                    strategy_names.append(strategy)
        
        plt.scatter(latencies, defense_effectiveness, s=100, alpha=0.7)
        
        for i, name in enumerate(strategy_names):
            plt.annotate(name, (latencies[i], defense_effectiveness[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Latency (ms)')
        plt.ylabel('Defense Effectiveness (AUC reduction)')
        plt.title('Privacy-Performance Trade-off')
        plt.grid(True, alpha=0.3)
        
    except FileNotFoundError:
        plt.text(0.5, 0.5, 'Attack results not available\nfor comparison', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('results/figures/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n=== Performance Summary ===")
    
    baseline_perf = perf_df[perf_df['strategy'] == 'none']
    baseline_latency = baseline_perf['mean_latency_ms'].mean()
    baseline_throughput = baseline_perf['throughput_samples_per_sec'].mean()
    
    print(f"Baseline performance: {baseline_latency:.3f} ms, {baseline_throughput:.0f} samples/sec")
    
    print("\nOverhead analysis:")
    for strategy in ['top_k_k1', 'top_k_k3', 'confidence_t0.3']:
        strategy_data = perf_df[perf_df['strategy'] == strategy]
        if len(strategy_data) > 0:
            strategy_latency = strategy_data['mean_latency_ms'].mean()
            overhead_pct = (strategy_latency - baseline_latency) / baseline_latency * 100
            print(f"  {strategy}: +{overhead_pct:.1f}% latency overhead")
    
    print(f"\n✓ Performance analysis saved to results/performance_analysis.csv")
    print(f"✓ Performance plots saved to results/figures/performance_analysis.png")
    
    return perf_df

if __name__ == "__main__":
    perf_df = computational_performance_analysis()