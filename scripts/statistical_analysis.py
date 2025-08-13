# statistical_analysis.py - Add statistical rigor to your results
import pandas as pd
import numpy as np
from scipy import stats
import json

def statistical_analysis():
    """
    Perform statistical significance testing on your results
    """
    # Load your results
    df = pd.read_csv('results/comprehensive_results.csv')
    
    print("=== Statistical Significance Analysis ===")
    
    # 1. Overfitting correlation analysis
    baseline_data = df[df['truncation_strategy'] == 'baseline']
    
    print("\n1. Overfitting vs Attack Success Correlation:")
    for attack in ['PCA', 'MPE', 'MLP']:
        attack_data = baseline_data[baseline_data['attack_method'] == attack]
        correlation, p_value = stats.pearsonr(attack_data['overfitting_level'], attack_data['auc'])
        print(f"   {attack}: r={correlation:.3f}, p={p_value:.3f}")
    
    # 2. Truncation effectiveness testing
    print("\n2. Truncation Effectiveness (Paired t-tests):")
    
    truncation_strategies = ['top_k_k1', 'top_k_k3', 'confidence_t0.3', 'confidence_t0.5']
    
    statistical_results = []
    
    for strategy in truncation_strategies:
        print(f"\n   {strategy}:")
        
        for attack in ['PCA', 'MPE', 'MLP']:
            # Get baseline and truncated AUCs for this attack
            baseline_aucs = df[(df['truncation_strategy'] == 'baseline') & 
                             (df['attack_method'] == attack)]['auc'].values
            truncated_aucs = df[(df['truncation_strategy'] == strategy) & 
                              (df['attack_method'] == attack)]['auc'].values
            
            if len(baseline_aucs) > 0 and len(truncated_aucs) > 0:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(baseline_aucs, truncated_aucs)
                
                # Effect size (Cohen's d)
                diff = baseline_aucs - truncated_aucs
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                
                # Mean reduction
                mean_reduction = np.mean(baseline_aucs) - np.mean(truncated_aucs)
                
                print(f"     {attack}: Δ={mean_reduction:.3f}, t={t_stat:.3f}, p={p_value:.3f}, d={cohens_d:.3f}")
                
                statistical_results.append({
                    'truncation_strategy': strategy,
                    'attack_method': attack,
                    'mean_baseline_auc': np.mean(baseline_aucs),
                    'mean_truncated_auc': np.mean(truncated_aucs),
                    'auc_reduction': mean_reduction,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                })
    
    # Save statistical results
    stats_df = pd.DataFrame(statistical_results)
    stats_df.to_csv('results/statistical_analysis.csv', index=False)
    
    print(f"\n✓ Statistical analysis saved to results/statistical_analysis.csv")
    
    # 3. Summary for paper
    print("\n3. Key Statistical Findings for Your Paper:")
    
    # Most effective defenses
    significant_reductions = stats_df[stats_df['significant'] == True]
    if len(significant_reductions) > 0:
        best_defense = significant_reductions.loc[significant_reductions['auc_reduction'].idxmax()]
        print(f"   Best defense: {best_defense['truncation_strategy']} vs {best_defense['attack_method']}")
        print(f"   AUC reduction: {best_defense['auc_reduction']:.3f} (p={best_defense['p_value']:.3f})")
    
    return stats_df

if __name__ == "__main__":
    stats_df = statistical_analysis()