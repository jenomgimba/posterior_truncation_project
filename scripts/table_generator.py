# table_generator.py - Generate LaTeX tables for your paper
import pandas as pd
import numpy as np

def generate_paper_tables():
    """
    Generate clean LaTeX tables for your IEEE paper
    """
    print("=== Generating Tables for Paper ===")
    
    # Load results
    df = pd.read_csv('results/comprehensive_results.csv')
    
    # Table 1: Model Characteristics and Baseline Performance
    print("\n1. Table I: Model Characteristics and Baseline Attack Performance")
    
    baseline_data = df[df['truncation_strategy'] == 'baseline']
    
    # Pivot to get attacks as columns
    table1_data = baseline_data.pivot_table(
        values='auc', 
        index=['model_name', 'overfitting_level'], 
        columns='attack_method', 
        aggfunc='mean'
    ).reset_index()
    
    # Calculate average AUC
    table1_data['Average'] = table1_data[['PCA', 'MPE', 'MLP']].mean(axis=1)
    
    # Clean model names
    table1_data['Model'] = table1_data['model_name'].str.replace('resnet18_', '').str.replace('_', ' ').str.title()
    
    # Generate LaTeX
    latex_table1 = """
\\begin{table}[htbp]
\\centering
\\caption{Model Characteristics and Baseline Attack Performance}
\\label{tab:baseline_performance}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Model} & \\textbf{Overfitting} & \\textbf{PCA} & \\textbf{MPE} & \\textbf{MLP} & \\textbf{Average} \\\\
\\hline
"""
    
    for _, row in table1_data.iterrows():
        latex_table1 += f"{row['Model']} & {row['overfitting_level']:.3f} & {row['PCA']:.3f} & {row['MPE']:.3f} & {row['MLP']:.3f} & {row['Average']:.3f} \\\\\n"
    
    latex_table1 += """\\hline
\\end{tabular}
\\end{table}
"""
    
    print(latex_table1)
    
    # Table 2: Best Defense Performance
    print("\n2. Table II: Truncation Defense Effectiveness")
    
    # Find best defense for each model-attack combination
    defense_data = df[df['truncation_strategy'] != 'baseline']
    
    best_defenses = []
    
    for model in df['model_name'].unique():
        for attack in ['PCA', 'MPE', 'MLP']:
            # Get baseline AUC
            baseline_auc = df[(df['model_name'] == model) & 
                            (df['attack_method'] == attack) & 
                            (df['truncation_strategy'] == 'baseline')]['auc'].iloc[0]
            
            # Find best defense
            model_attack_data = defense_data[(defense_data['model_name'] == model) & 
                                           (defense_data['attack_method'] == attack)]
            
            if len(model_attack_data) > 0:
                best_defense_row = model_attack_data.loc[model_attack_data['auc'].idxmin()]
                improvement = baseline_auc - best_defense_row['auc']
                
                best_defenses.append({
                    'Model': model.replace('resnet18_', '').replace('_', ' ').title(),
                    'Attack': attack,
                    'Baseline AUC': baseline_auc,
                    'Best Defense': best_defense_row['truncation_strategy'],
                    'Defense AUC': best_defense_row['auc'],
                    'Improvement': improvement
                })
    
    defense_df = pd.DataFrame(best_defenses)
    
    latex_table2 = """
\\begin{table}[htbp]
\\centering
\\caption{Best Truncation Defense Performance by Model and Attack}
\\label{tab:defense_effectiveness}
\\begin{tabular}{llcccc}
\\hline
\\textbf{Model} & \\textbf{Attack} & \\textbf{Baseline} & \\textbf{Best Defense} & \\textbf{Defended} & \\textbf{$\\Delta$AUC} \\\\
\\hline
"""
    
    for _, row in defense_df.iterrows():
        defense_name = row['Best Defense'].replace('_', ' ').replace('k', 'k=').replace('t', '$\\tau$=')
        latex_table2 += f"{row['Model']} & {row['Attack']} & {row['Baseline AUC']:.3f} & {defense_name} & {row['Defense AUC']:.3f} & {row['Improvement']:.3f} \\\\\n"
    
    latex_table2 += """\\hline
\\end{tabular}
\\end{table}
"""
    
    print(latex_table2)
    
    # Table 3: Statistical Significance Summary
    try:
        stats_df = pd.read_csv('results/statistical_analysis.csv')
        
        print("\n3. Table III: Statistical Significance of Defense Effectiveness")
        
        # Focus on significant results
        significant_results = stats_df[stats_df['significant'] == True]
        
        latex_table3 = """
\\begin{table}[htbp]
\\centering
\\caption{Statistical Significance of Truncation Defenses}
\\label{tab:statistical_significance}
\\begin{tabular}{llcccc}
\\hline
\\textbf{Defense Strategy} & \\textbf{Attack} & \\textbf{AUC Reduction} & \\textbf{t-statistic} & \\textbf{p-value} & \\textbf{Effect Size} \\\\
\\hline
"""
        
        for _, row in significant_results.iterrows():
            strategy_name = row['truncation_strategy'].replace('_', ' ')
            latex_table3 += f"{strategy_name} & {row['attack_method']} & {row['auc_reduction']:.3f} & {row['t_statistic']:.2f} & {row['p_value']:.3f} & {row['cohens_d']:.2f} \\\\\n"
        
        latex_table3 += """\\hline
\\multicolumn{6}{l}{\\footnotesize Significant results (p < 0.05) only. Effect size: Cohen's d.} \\\\
\\end{tabular}
\\end{table}
"""
        
        print(latex_table3)
        
    except FileNotFoundError:
        print("Statistical analysis not available yet - run statistical_analysis.py first")
    
    # Save all tables to file
    with open('results/latex_tables.txt', 'w') as f:
        f.write("LaTeX Tables for Paper\n")
        f.write("=" * 50 + "\n\n")
        f.write("Table I: Model Characteristics and Baseline Performance\n")
        f.write(latex_table1)
        f.write("\n" + "=" * 50 + "\n\n")
        f.write("Table II: Defense Effectiveness\n")
        f.write(latex_table2)
        f.write("\n" + "=" * 50 + "\n\n")
        try:
            f.write("Table III: Statistical Significance\n")
            f.write(latex_table3)
        except:
            f.write("Table III: Statistical analysis not available\n")
    
    print(f"\n✓ All LaTeX tables saved to results/latex_tables.txt")
    print("✓ Copy-paste ready for your IEEE paper!")
    
    return table1_data, defense_df

if __name__ == "__main__":
    table1, table2 = generate_paper_tables()