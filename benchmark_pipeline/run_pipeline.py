"""
Main Pipeline Runner for GO Benchmark Analysis

This script orchestrates the entire analysis workflow from preprocessed data
to final results and visualizations.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from benchmark_analysis import BenchmarkAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Complete analysis pipeline orchestrator
    """
    
    def __init__(self, input_file, output_dir, config=None):
        """
        Initialize pipeline
        
        Parameters:
        -----------
        input_file : str
            Path to preprocessed/combined TSV file
        output_dir : str
            Directory for saving all outputs
        config : dict, optional
            Configuration parameters
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Create output subdirectories
        self.figures_dir = self.output_dir / 'figures'
        self.reports_dir = self.output_dir / 'reports'
        self.data_dir = self.output_dir / 'data'
        
        for directory in [self.figures_dir, self.reports_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzer
        self.analyzer = None
        
    def validate_input(self):
        """
        Validate that input file exists and is readable
        """
        logger.info("="*70)
        logger.info("VALIDATING INPUT")
        logger.info("="*70)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        logger.info(f"✓ Input file exists: {self.input_file}")
        logger.info(f"  Size: {self.input_file.stat().st_size / 1024:.2f} KB")
        
    def run_analysis(self):
        """
        Run the complete benchmark analysis
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING BENCHMARK ANALYSIS")
        logger.info("="*70 + "\n")
        
        # Initialize analyzer
        self.analyzer = BenchmarkAnalyzer(str(self.input_file))
        
        # Calculate all metrics
        logger.info("Step 1/11: Calculating Balanced Accuracy...")
        self.analyzer.calculate_balanced_accuracy()
        
        logger.info("\nStep 2/11: Calculating MCC...")
        self.analyzer.calculate_mcc()
        
        logger.info("\nStep 3/11: Calculating Efficiency Metrics...")
        self.analyzer.calculate_efficiency_metrics()
        
        logger.info("\nStep 4/11: Aggregating by Algorithm...")
        self.analyzer.aggregate_by_algorithm()
        
        logger.info("\nStep 5/11: Analyzing Term Stratification...")
        self.analyzer.analyze_term_stratification()
        
        logger.info("\nStep 6/11: Calculating Error Correlation...")
        self.analyzer.calculate_error_correlation()
        
        # Generate visualizations
        logger.info("\nStep 7/11: Creating Efficiency Frontier Plot...")
        self.analyzer.visualize_efficiency_frontier(
            self.figures_dir / 'efficiency_frontier.png'
        )
        
        logger.info("\nStep 8/11: Creating Algorithm Comparison Plot...")
        self.analyzer.visualize_algorithm_comparison(
            self.figures_dir / 'algorithm_comparison.png'
        )
        
        logger.info("\nStep 9/11: Creating Correlation Heatmap...")
        self.analyzer.visualize_correlation_heatmap(
            self.figures_dir / 'correlation_heatmap.png'
        )
        
        # Identify best baseline
        logger.info("\nStep 10/11: Identifying Best Baseline...")
        self.analyzer.identify_best_baseline()
        
        # Export results
        logger.info("\nStep 11/11: Exporting Results...")
        self.analyzer.export_summary_report(
            self.reports_dir / 'benchmark_summary_report.csv'
        )
        
        # Save enhanced data
        self.analyzer.df.to_csv(
            self.data_dir / 'enhanced_benchmark_data.csv',
            index=False
        )
        
    def generate_summary_report(self):
        """
        Generate a text summary report
        """
        logger.info("\n" + "="*70)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("="*70)
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("GO BENCHMARK ANALYSIS SUMMARY REPORT")
        report_lines.append("="*70)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Input File: {self.input_file.name}")
        report_lines.append(f"\n{'-'*70}")
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-"*70)
        
        df = self.analyzer.df
        report_lines.append(f"Total Observations: {len(df)}")
        
        # Find column names
        algo_col = next((col for col in df.columns if 'algo' in col.lower()), None)
        go_col = next((col for col in df.columns if col.lower().startswith('go') and 'level' not in col.lower()), None)
        
        if algo_col:
            report_lines.append(f"Algorithms: {df[algo_col].nunique()}")
            for algo in sorted(df[algo_col].unique()):
                count = (df[algo_col] == algo).sum()
                report_lines.append(f"  - {algo}: {count} observations")
        
        if go_col:
            report_lines.append(f"\nGO Terms: {df[go_col].nunique()}")
            report_lines.append(f"Terms: {', '.join(sorted(df[go_col].unique()))}")
        
        if 'GO_Level' in df.columns:
            report_lines.append(f"\nGO Levels: {df['GO_Level'].nunique()}")
            for level in sorted(df['GO_Level'].unique()):
                count = (df['GO_Level'] == level).sum()
                report_lines.append(f"  - {level}: {count} observations")
        
        # Performance summary
        if self.analyzer.summary_stats is not None:
            report_lines.append(f"\n{'-'*70}")
            report_lines.append("PERFORMANCE SUMMARY")
            report_lines.append("-"*70)
            
            # Find F1 mean column
            f1_cols = [col for col in self.analyzer.summary_stats.columns if 'f1' in str(col).lower()]
            if f1_cols:
                f1_mean_col = [col for col in f1_cols if 'mean' in str(col)][0]
                
                report_lines.append("\nMean F1 Scores by Algorithm:")
                for algo in self.analyzer.summary_stats.index:
                    f1_mean = self.analyzer.summary_stats.loc[algo, f1_mean_col]
                    report_lines.append(f"  {algo}: {f1_mean:.4f}")
                
                # Best algorithm
                best_algo = self.analyzer.summary_stats[f1_mean_col].idxmax()
                best_f1 = self.analyzer.summary_stats.loc[best_algo, f1_mean_col]
                report_lines.append(f"\n🏆 Best Baseline: {best_algo} (F1 = {best_f1:.4f})")
        
        # Output files
        report_lines.append(f"\n{'-'*70}")
        report_lines.append("OUTPUT FILES")
        report_lines.append("-"*70)
        report_lines.append("\nFigures:")
        report_lines.append(f"  - {self.figures_dir / 'efficiency_frontier.png'}")
        report_lines.append(f"  - {self.figures_dir / 'algorithm_comparison.png'}")
        report_lines.append(f"  - {self.figures_dir / 'correlation_heatmap.png'}")
        report_lines.append("\nData & Reports:")
        report_lines.append(f"  - {self.data_dir / 'enhanced_benchmark_data.csv'}")
        report_lines.append(f"  - {self.reports_dir / 'benchmark_summary_report.csv'}")
        
        report_lines.append("\n" + "="*70)
        
        # Save to file
        report_text = '\n'.join(report_lines)
        report_file = self.reports_dir / 'analysis_summary.txt'
        report_file.write_text(report_text)
        
        logger.info(f"✓ Summary report saved to: {report_file}")
        
        # Also print to console
        print("\n" + report_text)
    
    def run(self):
        """
        Execute the complete pipeline
        """
        start_time = datetime.now()
        
        try:
            # Validate input
            self.validate_input()
            
            # Run analysis
            self.run_analysis()
            
            # Generate summary
            self.generate_summary_report()
            
            # Success
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"Total time: {elapsed:.1f} seconds")
            logger.info(f"Results saved to: {self.output_dir.absolute()}")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Command-line interface for pipeline
    """
    parser = argparse.ArgumentParser(
        description='Run complete GO benchmark analysis pipeline'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input TSV file (preprocessed/combined data)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run pipeline
    pipeline = AnalysisPipeline(args.input, args.output)
    success = pipeline.run()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
