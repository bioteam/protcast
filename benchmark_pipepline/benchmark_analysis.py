"""
Benchmark Performance Analysis for Protein Function Prediction
================================================================
This script analyzes TSV data comparing traditional feature extraction methods
(AAC, APAAC, CKSAAP, etc.) for protein GO term prediction.

Input: TSV file with columns including:
    - Algorithm, GO_Term, Sensitivity, Specificity, F1_Score, 
      Vector_Length, Elapsed_Time, etc.

Output: Comprehensive statistical analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BenchmarkAnalyzer:
    """Comprehensive analyzer for ML model benchmarking"""
    
    def __init__(self, tsv_path):
        """
        Initialize analyzer with TSV data
        
        Parameters:
        -----------
        tsv_path : str
            Path to the TSV file containing benchmark data
        """
        print(f"Loading data from {tsv_path}...")
        self.df = pd.read_csv(tsv_path, sep='\t')
        print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        # Store results
        self.summary_stats = None
        self.correlation_matrix = None
        
    def calculate_balanced_accuracy(self):
        """
        Calculate balanced accuracy for each row
        Balanced Accuracy = (Sensitivity + Specificity) / 2
        """
        print("\n" + "="*60)
        print("1. CALCULATING BALANCED ACCURACY")
        print("="*60)
        
        required_cols = ['Sensitivity', 'Specificity']
        
        # Check if columns exist (case-insensitive)
        col_mapping = {}
        for req_col in required_cols:
            found = False
            for col in self.df.columns:
                if col.lower() == req_col.lower():
                    col_mapping[req_col] = col
                    found = True
                    break
            if not found:
                print(f"Warning: Column '{req_col}' not found in data")
                return
        
        # Calculate balanced accuracy
        self.df['Balanced_Accuracy'] = (
            self.df[col_mapping['Sensitivity']] + 
            self.df[col_mapping['Specificity']]
        ) / 2
        
        print(f"Balanced Accuracy calculated for all {len(self.df)} rows")
        print("\nSample balanced accuracies:")
        print(self.df[['Balanced_Accuracy']].describe())
        
    def calculate_mcc(self):
        """
        Calculate Matthews Correlation Coefficient if TP, TN, FP, FN are available
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        """
        print("\n" + "="*60)
        print("2. CALCULATING MATTHEWS CORRELATION COEFFICIENT (MCC)")
        print("="*60)
        
        required = ['TP', 'TN', 'FP', 'FN']
        
        # Check if confusion matrix values exist
        has_all = all(any(col.upper() == req for col in self.df.columns) for req in required)
        
        if not has_all:
            print("Note: TP, TN, FP, FN columns not found.")
            print("MCC requires raw confusion matrix values.")
            print("If you have Sensitivity, Specificity, and total counts, we can estimate MCC.")
            return
        
        # Find the actual column names
        col_map = {}
        for req in required:
            for col in self.df.columns:
                if col.upper() == req:
                    col_map[req] = col
                    break
        
        # Calculate MCC
        tp = self.df[col_map['TP']]
        tn = self.df[col_map['TN']]
        fp = self.df[col_map['FP']]
        fn = self.df[col_map['FN']]
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        # Handle division by zero
        self.df['MCC'] = np.where(denominator == 0, 0, numerator / denominator)
        
        print(f"MCC calculated for all {len(self.df)} rows")
        print("\nMCC statistics:")
        print(self.df['MCC'].describe())
        
    def calculate_efficiency_metrics(self):
        """
        Calculate efficiency metrics: F1 per dimension and F1 per millisecond
        """
        print("\n" + "="*60)
        print("3. CALCULATING EFFICIENCY METRICS")
        print("="*60)
        
        # Find column names (case-insensitive)
        f1_col = next((col for col in self.df.columns if 'f1' in col.lower()), None)
        vec_col = next((col for col in self.df.columns if 'vector' in col.lower() and 'length' in col.lower()), None)
        time_col = next((col for col in self.df.columns if 'time' in col.lower() or 'elapsed' in col.lower()), None)
        
        if f1_col is None:
            print("Warning: F1 Score column not found")
            return
        
        if vec_col:
            self.df['F1_per_Dimension'] = self.df[f1_col] / (self.df[vec_col] + 1e-10)
            print(f"✓ F1 per Dimension calculated")
        else:
            print("Note: Vector Length column not found, skipping F1_per_Dimension")
        
        if time_col:
            # Convert time to seconds if in milliseconds
            time_values = self.df[time_col]
            if time_values.max() > 1000:  # Likely in milliseconds
                time_seconds = time_values / 1000
            else:
                time_seconds = time_values
            
            self.df['F1_per_Second'] = self.df[f1_col] / (time_seconds + 1e-10)
            print(f"✓ F1 per Second calculated")
        else:
            print("Note: Time column not found, skipping F1_per_Second")
        
        print("\nEfficiency metrics summary:")
        if 'F1_per_Dimension' in self.df.columns:
            print(f"F1_per_Dimension: {self.df['F1_per_Dimension'].describe()}")
        if 'F1_per_Second' in self.df.columns:
            print(f"F1_per_Second: {self.df['F1_per_Second'].describe()}")
    
    def aggregate_by_algorithm(self):
        """
        Aggregate performance metrics by algorithm (macro-averaging)
        """
        print("\n" + "="*60)
        print("4. AGGREGATING BY ALGORITHM (MACRO-AVERAGING)")
        print("="*60)
        
        # Find algorithm column
        algo_col = next((col for col in self.df.columns if 'algo' in col.lower()), None)
        if algo_col is None:
            print("Warning: Algorithm column not found")
            return None
        
        # Metrics to aggregate
        metrics_to_agg = []
        for metric in ['F1', 'Balanced_Accuracy', 'MCC', 'F1_per_Dimension', 'F1_per_Second']:
            col = next((c for c in self.df.columns if metric.lower() in c.lower()), None)
            if col:
                metrics_to_agg.append(col)
        
        if not metrics_to_agg:
            print("Warning: No metrics found to aggregate")
            return None
        
        # Group by algorithm and calculate statistics
        self.summary_stats = self.df.groupby(algo_col)[metrics_to_agg].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        
        print("\nSummary Statistics by Algorithm:")
        print(self.summary_stats)
        
        return self.summary_stats
    
    def analyze_term_stratification(self):
        """
        Analyze performance stratified by GO term characteristics
        """
        print("\n" + "="*60)
        print("5. TERM-CENTRIC STRATIFICATION ANALYSIS")
        print("="*60)
        
        # Find GO term column
        go_col = next((col for col in self.df.columns if col.lower().startswith('go') and 'level' not in col.lower()), None)
        if go_col is None:
            print("Note: GO Term column not found")
            return
        
        print(f"\nUnique GO terms: {self.df[go_col].nunique()}")
        print(f"GO terms in dataset: {sorted(self.df[go_col].unique())}")
        
        # Calculate per-term average performance
        f1_col = next((col for col in self.df.columns if 'f1' in col.lower()), None)
        if f1_col:
            term_performance = self.df.groupby(go_col)[f1_col].agg(['mean', 'std', 'count'])
            term_performance = term_performance.sort_values('mean')
            
            print("\nPer-term average F1 scores:")
            print(term_performance)
            
            # Identify "head" and "tail" terms
            median_f1 = term_performance['mean'].median()
            head_terms = term_performance[term_performance['mean'] >= median_f1].index
            tail_terms = term_performance[term_performance['mean'] < median_f1].index
            
            print(f"\n'Head' terms (F1 >= {median_f1:.3f}): {len(head_terms)}")
            print(f"'Tail' terms (F1 < {median_f1:.3f}): {len(tail_terms)}")
    
    def calculate_error_correlation(self):
        """
        Calculate correlation matrix of F1 scores across algorithms
        """
        print("\n" + "="*60)
        print("6. ERROR CORRELATION ANALYSIS")
        print("="*60)
        
        # Find algorithm and F1 columns
        algo_col = next((col for col in self.df.columns if 'algo' in col.lower()), None)
        f1_col = next((col for col in self.df.columns if 'f1' in col.lower()), None)
        go_col = next((col for col in self.df.columns if col.lower().startswith('go') and 'level' not in col.lower()), None)
        
        if not all([algo_col, f1_col, go_col]):
            print("Warning: Required columns not found for correlation analysis")
            return None
        
        # Pivot to create algorithm x GO term matrix
        pivot_table = self.df.pivot_table(
            values=f1_col,
            index=go_col,
            columns=algo_col,
            aggfunc='mean'
        )
        
        # Calculate correlation
        self.correlation_matrix = pivot_table.corr(method='spearman')
        
        print("\nAlgorithm Performance Correlation Matrix (Spearman):")
        print(self.correlation_matrix.round(3))
        
        return self.correlation_matrix
    
    def visualize_efficiency_frontier(self, save_path='efficiency_frontier.png'):
        """
        Create scatter plot of Performance vs. Computational Cost
        """
        print("\n" + "="*60)
        print("7. VISUALIZING EFFICIENCY FRONTIER")
        print("="*60)
        
        # Find relevant columns
        algo_col = next((col for col in self.df.columns if 'algo' in col.lower()), None)
        f1_col = next((col for col in self.df.columns if 'f1' in col.lower()), None)
        time_col = next((col for col in self.df.columns if 'time' in col.lower() or 'elapsed' in col.lower()), None)
        vec_col = next((col for col in self.df.columns if 'vector' in col.lower() and 'length' in col.lower()), None)
        
        if not all([algo_col, f1_col]):
            print("Warning: Cannot create efficiency frontier - missing required columns")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: F1 vs Vector Length
        if vec_col:
            for algo in self.df[algo_col].unique():
                algo_data = self.df[self.df[algo_col] == algo]
                axes[0].scatter(
                    algo_data[vec_col],
                    algo_data[f1_col],
                    label=algo,
                    alpha=0.7,
                    s=100
                )
            
            axes[0].set_xlabel('Vector Length (dimensions)', fontsize=12)
            axes[0].set_ylabel('F1 Score', fontsize=12)
            axes[0].set_title('Efficiency Frontier: F1 vs Vector Dimensionality', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: F1 vs Elapsed Time
        if time_col:
            for algo in self.df[algo_col].unique():
                algo_data = self.df[self.df[algo_col] == algo]
                time_values = algo_data[time_col]
                
                # Use log scale if values span multiple orders of magnitude
                if time_values.max() / (time_values.min() + 1e-10) > 100:
                    axes[1].set_xscale('log')
                
                axes[1].scatter(
                    time_values,
                    algo_data[f1_col],
                    label=algo,
                    alpha=0.7,
                    s=100
                )
            
            axes[1].set_xlabel('Elapsed Time (ms)', fontsize=12)
            axes[1].set_ylabel('F1 Score', fontsize=12)
            axes[1].set_title('Efficiency Frontier: F1 vs Computation Time', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved efficiency frontier plot to {save_path}")
        plt.close()
    
    def visualize_algorithm_comparison(self, save_path='algorithm_comparison.png'):
        """
        Create boxplot comparing F1 distribution across algorithms
        """
        print("\n" + "="*60)
        print("8. VISUALIZING ALGORITHM COMPARISON")
        print("="*60)
        
        algo_col = next((col for col in self.df.columns if 'algo' in col.lower()), None)
        f1_col = next((col for col in self.df.columns if 'f1' in col.lower()), None)
        
        if not all([algo_col, f1_col]):
            print("Warning: Cannot create comparison plot - missing required columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create boxplot
        self.df.boxplot(
            column=f1_col,
            by=algo_col,
            ax=ax,
            patch_artist=True,
            showmeans=True
        )
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('F1 Score Distribution Across GO Terms by Algorithm', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        
        # Rotate x labels if needed
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved algorithm comparison plot to {save_path}")
        plt.close()
    
    def visualize_correlation_heatmap(self, save_path='correlation_heatmap.png'):
        """
        Create heatmap of algorithm error correlations
        """
        if self.correlation_matrix is None:
            print("Note: Correlation matrix not calculated yet")
            return
        
        print("\n" + "="*60)
        print("9. VISUALIZING ERROR CORRELATION HEATMAP")
        print("="*60)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Algorithm Performance Correlation\n(Spearman correlation of F1 scores across GO terms)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved correlation heatmap to {save_path}")
        plt.close()
    
    def identify_best_baseline(self):
        """
        Identify the single best baseline algorithm
        """
        print("\n" + "="*60)
        print("10. IDENTIFYING BEST BASELINE")
        print("="*60)
        
        if self.summary_stats is None:
            print("Note: Run aggregate_by_algorithm() first")
            return
        
        # Find F1 column in summary stats
        f1_cols = [col for col in self.summary_stats.columns if 'f1' in str(col).lower()]
        
        if not f1_cols:
            print("Warning: No F1 scores found in summary statistics")
            return
        
        # Get mean F1 for each algorithm
        f1_mean_col = [col for col in f1_cols if 'mean' in str(col)][0]
        best_algo = self.summary_stats[f1_mean_col].idxmax()
        best_f1 = self.summary_stats.loc[best_algo, f1_mean_col]
        
        print(f"\n🏆 BEST BASELINE ALGORITHM: {best_algo}")
        print(f"   Mean F1 Score: {best_f1:.4f}")
        print(f"\n   This is the 'Hard Baseline' your ESM model must beat!")
        
        # Show full statistics for best algorithm
        print(f"\n   Full statistics for {best_algo}:")
        print(self.summary_stats.loc[best_algo])
    
    def export_summary_report(self, output_path='benchmark_summary_report.csv'):
        """
        Export comprehensive summary report
        """
        print("\n" + "="*60)
        print("11. EXPORTING SUMMARY REPORT")
        print("="*60)
        
        if self.summary_stats is not None:
            # Flatten multi-index columns
            self.summary_stats.columns = ['_'.join(col).strip() for col in self.summary_stats.columns.values]
            self.summary_stats.to_csv(output_path)
            print(f"✓ Saved summary report to {output_path}")
        else:
            print("Note: No summary statistics to export")
    
    def run_full_analysis(self, output_dir='./output'):
        """
        Run complete analysis pipeline
        """
        print("\n" + "="*70)
        print("BENCHMARK ANALYSIS PIPELINE")
        print("="*70)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"\nOutput directory: {output_dir.absolute()}")
        
        # Run all analyses
        self.calculate_balanced_accuracy()
        self.calculate_mcc()
        self.calculate_efficiency_metrics()
        self.aggregate_by_algorithm()
        self.analyze_term_stratification()
        self.calculate_error_correlation()
        
        # Generate visualizations
        self.visualize_efficiency_frontier(output_dir / 'efficiency_frontier.png')
        self.visualize_algorithm_comparison(output_dir / 'algorithm_comparison.png')
        self.visualize_correlation_heatmap(output_dir / 'correlation_heatmap.png')
        
        # Identify best baseline
        self.identify_best_baseline()
        
        # Export results
        self.export_summary_report(output_dir / 'benchmark_summary_report.csv')
        
        # Save the enhanced dataframe
        self.df.to_csv(output_dir / 'enhanced_benchmark_data.csv', index=False)
        print(f"\n✓ Saved enhanced data with calculated metrics to enhanced_benchmark_data.csv")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {output_dir.absolute()}")
        print("\nGenerated files:")
        print("  1. efficiency_frontier.png - Performance vs cost scatter plots")
        print("  2. algorithm_comparison.png - F1 score distributions by algorithm")
        print("  3. correlation_heatmap.png - Algorithm error correlation matrix")
        print("  4. benchmark_summary_report.csv - Statistical summary by algorithm")
        print("  5. enhanced_benchmark_data.csv - Original data + calculated metrics")


def main():
    """
    Main execution function
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        BENCHMARK PERFORMANCE ANALYSIS FOR PROTEIN FUNCTION       ║
    ║                      PREDICTION MODELS                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Example usage
    print("USAGE:")
    print("-" * 60)
    print("analyzer = BenchmarkAnalyzer('your_data.tsv')")
    print("analyzer.run_full_analysis(output_dir='./results')")
    print("-" * 60)
    print("\nOr run individual analyses:")
    print("  analyzer.calculate_balanced_accuracy()")
    print("  analyzer.aggregate_by_algorithm()")
    print("  analyzer.visualize_efficiency_frontier()")
    print("  etc.")
    print("\n")
    
    # If you want to run directly, uncomment and modify:
    # analyzer = BenchmarkAnalyzer('benchmark_data.tsv')
    # analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
