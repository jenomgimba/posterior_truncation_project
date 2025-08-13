# error_analysis.py - Comprehensive manual error analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def comprehensive_error_analysis():
    """
    Perform detailed error analysis required by your project specification
    """
    print("=== Comprehensive Error Analysis ===")
    
    df = pd.read_csv('results/comprehensive_results.csv')
    
    # 1. False Positive Analysis (Non-members identified as members)
    print("\n1. Attack Failure Modes Analysis:")
    
    failure_analysis = []
    
    # Analyze when attacks fail (AUC close to 0.5)
    baseline_data = df[df['truncation_strategy'] == 'baseline']
    
    for _, row in baseline_data.iterrows():
        if row['auc'] < 0.55:  # Poor attack performance
            failure_analysis.append({
                'model': row['model_name'],
                'attack': row['attack_method'],
                'auc': row['auc'],
                'overfitting': row['overfitting_level'],
                'failure_type': 'Low attack success'
            })
    
    # 2. Truncation Failure Analysis
    print("\n2. Truncation Failure Modes:")
    
    truncation_failures = []
    
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        
        for attack in ['PCA', 'MPE', 'MLP']:
            baseline_auc = model_data[(model_data['attack_method'] == attack) & 
                                    (model_data['truncation_strategy'] == 'baseline')]['auc'].iloc[0]
            
            # Find cases where truncation doesn't help or makes things worse
            attack_data = model_data[model_data['attack_method'] == attack]
            
            for _, row in attack_data.iterrows():
                if row['truncation_strategy'] != 'baseline':
                    auc_change = baseline_auc - row['auc']
                    
                    if auc_change < 0.01:  # No significant improvement
                        truncation_failures.append({
                            'model': model,
                            'attack': attack,
                            'truncation': row['truncation_strategy'],
                            'baseline_auc': baseline_auc,
                            'truncated_auc': row['auc'],
                            'improvement': auc_change,
                            'failure_reason': 'Insufficient defense' if auc_change >= 0 else 'Defense backfire'
                        })
    
    # 3. Model-specific vulnerabilities
    print("\n3. Model-Specific Vulnerability Patterns:")
    
    vulnerability_patterns = []
    
    for model in df['model_name'].unique():
        model_baseline = df[(df['model_name'] == model) & (df['truncation_strategy'] == 'baseline')]
        
        # Find which attack is most effective for each model
        avg_auc = model_baseline.groupby('attack_method')['auc'].mean()
        most_vulnerable_attack = avg_auc.idxmax()
        highest_auc = avg_auc.max()
        
        # Find best defense for this model
        model_data = df[df['model_name'] == model]
        defense_effectiveness = []
        
        for strategy in model_data['truncation_strategy'].unique():
            if strategy != 'baseline':
                strategy_data = model_data[model_data['truncation_strategy'] == strategy]
                avg_defense_auc = strategy_data['auc'].mean()
                baseline_avg = model_baseline['auc'].mean()
                improvement = baseline_avg - avg_defense_auc
                
                defense_effectiveness.append({
                    'strategy': strategy,
                    'improvement': improvement
                })
        
        if defense_effectiveness:
            best_defense = max(defense_effectiveness, key=lambda x: x['improvement'])
            
            vulnerability_patterns.append({
                'model': model,
                'overfitting': model_baseline['overfitting_level'].iloc[0],
                'most_vulnerable_to': most_vulnerable_attack,
                'max_vulnerability_auc': highest_auc,
                'best_defense': best_defense['strategy'],
                'defense_improvement': best_defense['improvement']
            })
    
    # 4. Attack Method Comparative Analysis
    print("\n4. Attack Method Effectiveness Comparison:")
    
    attack_comparison = []
    baseline_data = df[df['truncation_strategy'] == 'baseline']
    
    for attack in ['PCA', 'MPE', 'MLP']:
        attack_data = baseline_data[baseline_data['attack_method'] == attack]
        
        attack_comparison.append({
            'attack_method': attack,
            'mean_auc': attack_data['auc'].mean(),
            'std_auc': attack_data['auc'].std(),
            'min_auc': attack_data['auc'].min(),
            'max_auc': attack_data['auc'].max(),
            'correlation_with_overfitting': attack_data['auc'].corr(attack_data['overfitting_level'])
        })
    
    # 5. Save comprehensive error analysis
    error_analysis_results = {
        'failure_analysis': failure_analysis,
        'truncation_failures': truncation_failures,
        'vulnerability_patterns': vulnerability_patterns,
        'attack_comparison': attack_comparison
    }
    
    with open('results/error_analysis.json', 'w') as f:
        json.dump(error_analysis_results, f, indent=2)
    
    # Create summary report
    print("\n=== Error Analysis Summary for Your Paper ===")
    
    print(f"\n📊 Attack Effectiveness Ranking:")
    attack_comp_df = pd.DataFrame(attack_comparison)
    attack_comp_df = attack_comp_df.sort_values('mean_auc', ascending=False)
    
    for _, row in attack_comp_df.iterrows():
        print(f"   {row['attack_method']}: {row['mean_auc']:.3f} ± {row['std_auc']:.3f} AUC")
    
    print(f"\n🎯 Most Vulnerable Models:")
    vuln_df = pd.DataFrame(vulnerability_patterns)
    vuln_df = vuln_df.sort_values('max_vulnerability_auc', ascending=False)
    
    for _, row in vuln_df.head(3).iterrows():
        print(f"   {row['model']}: {row['max_vulnerability_auc']:.3f} AUC ({row['most_vulnerable_to']})")
    
    print(f"\n🛡️ Most Effective Defenses:")
    best_defenses = vuln_df.sort_values('defense_improvement', ascending=False)
    
    for _, row in best_defenses.head(3).iterrows():
        print(f"   {row['model']}: {row['best_defense']} (Δ={row['defense_improvement']:.3f})")
    
    print(f"\n⚠️ Truncation Failure Cases:")
    if truncation_failures:
        failure_df = pd.DataFrame(truncation_failures)
        print(f"   {len(failure_df)} cases where truncation provided <0.01 AUC improvement")
        
        # Group by failure reason
        failure_counts = failure_df['failure_reason'].value_counts()
        for reason, count in failure_counts.items():
            print(f"   - {reason}: {count} cases")
    else:
        print("   ✓ No significant truncation failures detected")
    
    print(f"\n✓ Error analysis saved to results/error_analysis.json")
    
    return error_analysis_results

if __name__ == "__main__":
    error_analysis = comprehensive_error_analysis()