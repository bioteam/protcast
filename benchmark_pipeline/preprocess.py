"""
Data Preprocessing for GO Term Benchmark Data

This script handles concatenating multiple TSV files from different levels
of the Molecular Function GO DAG and prepares them for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GODataPreprocessor:
    """
    Preprocessor for GO term benchmark data from multiple DAG levels
    """
    
    def __init__(self, input_dir, output_file='combined_benchmark_data.tsv'):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        input_dir : str
            Directory containing TSV files from different GO levels
        output_file : str
            Path to save the combined output file
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.dataframes = []
        self.combined_df = None
        
    def find_tsv_files(self, pattern='*.tsv'):
        """
        Find all TSV files in the input directory
        
        Parameters:
        -----------
        pattern : str
            Glob pattern to match TSV files (default: '*.tsv')
        
        Returns:
        --------
        list : List of Path objects for TSV files
        """
        tsv_files = sorted(self.input_dir.glob(pattern))
        logger.info(f"Found {len(tsv_files)} TSV files in {self.input_dir}")
        
        if not tsv_files:
            logger.warning(f"No TSV files found in {self.input_dir}")
            # Also check for .txt files
            tsv_files = sorted(self.input_dir.glob('*.txt'))
            if tsv_files:
                logger.info(f"Found {len(tsv_files)} .txt files instead")
        
        for f in tsv_files:
            logger.info(f"  - {f.name}")
        
        return tsv_files
    
    def extract_go_level(self, filename):
        """
        Extract GO DAG level from filename
        
        Expected formats:
        - level_3.tsv
        - go_level3_results.tsv
        - molecular_function_L3.tsv
        
        Parameters:
        -----------
        filename : str
            Name of the file
        
        Returns:
        --------
        str or None : Extracted level identifier
        """
        import re
        
        # Try different patterns
        patterns = [
            r'level[_\s]?(\d+)',  # level_3, level3
            r'L(\d+)',             # L3
            r'lv(\d+)',            # lv3
            r'depth[_\s]?(\d+)',   # depth_3
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return f"Level_{match.group(1)}"
        
        # If no level found, use filename stem
        return Path(filename).stem
    
    def load_and_tag_file(self, filepath):
        """
        Load a TSV file and add a GO_Level column
        
        Parameters:
        -----------
        filepath : Path
            Path to the TSV file
        
        Returns:
        --------
        pd.DataFrame : Loaded dataframe with GO_Level column
        """
        logger.info(f"Loading {filepath.name}...")
        
        try:
            # Try reading with tab separator
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            logger.warning(f"Failed to read {filepath.name} with tab separator, trying comma")
            try:
                df = pd.read_csv(filepath, sep=',')
            except Exception as e2:
                logger.error(f"Failed to read {filepath.name}: {e2}")
                return None
        
        # Add GO level information
        go_level = self.extract_go_level(filepath.name)
        df['GO_Level'] = go_level
        df['Source_File'] = filepath.name
        
        logger.info(f"  Loaded {len(df)} rows from {go_level}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        return df
    
    def validate_schema(self, df, filepath):
        """
        Validate that dataframe has required columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to validate
        filepath : Path
            Source file path for error messages
        
        Returns:
        --------
        bool : True if valid, False otherwise
        """
        # Check for at least some key columns (flexible matching)
        required_patterns = ['algo', 'go', 'f1']  # Must have algo, go term, and F1
        
        df_cols_lower = [col.lower() for col in df.columns]
        
        for pattern in required_patterns:
            if not any(pattern in col for col in df_cols_lower):
                logger.warning(f"{filepath.name} missing column matching '{pattern}'")
                return False
        
        return True
    
    def standardize_column_names(self, df):
        """
        Standardize column names across different files
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to standardize
        
        Returns:
        --------
        pd.DataFrame : Dataframe with standardized column names
        """
        # Create a mapping of variations to standard names
        column_mapping = {
            # Algorithm variations
            'algo': 'Algorithm',
            'algorithm': 'Algorithm',
            'method': 'Algorithm',
            'model': 'Algorithm',
            
            # GO term variations
            'go': 'GO_Term',
            'goterm': 'GO_Term',
            'go_term': 'GO_Term',
            'term': 'GO_Term',
            'go_id': 'GO_Term',
            'type': 'GO_Term',
            
            # Performance metrics
            'f1': 'F1_Score',
            'f1_score': 'F1_Score',
            'f1score': 'F1_Score',
            'f_score': 'F1_Score',
            
            'sensitivity': 'Sensitivity',
            'recall': 'Sensitivity',
            'tpr': 'Sensitivity',
            
            'specificity': 'Specificity',
            'tnr': 'Specificity',
            
            # Efficiency metrics
            'vector_length': 'Vector_Length',
            'vector_len': 'Vector_Length',
            'dimensions': 'Vector_Length',
            'dim': 'Vector_Length',
            'feature_dim': 'Vector_Length',
            
            'elapsed_time': 'Elapsed_Time',
            'time': 'Elapsed_Time',
            'runtime': 'Elapsed_Time',
            'duration': 'Elapsed_Time',
            'time_ms': 'Elapsed_Time',
            
            # Confusion matrix
            'tp': 'TP',
            'true_positive': 'TP',
            'tn': 'TN',
            'true_negative': 'TN',
            'fp': 'FP',
            'false_positive': 'FP',
            'fn': 'FN',
            'false_negative': 'FN',
        }
        
        # Rename columns
        new_columns = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if col_lower in column_mapping:
                new_columns[col] = column_mapping[col_lower]
        
        if new_columns:
            df = df.rename(columns=new_columns)
            logger.info(f"  Standardized {len(new_columns)} column names")
        
        return df
    
    def concatenate_files(self, tsv_files):
        """
        Load and concatenate all TSV files
        
        Parameters:
        -----------
        tsv_files : list
            List of Path objects for TSV files
        
        Returns:
        --------
        pd.DataFrame : Combined dataframe
        """
        logger.info("\n" + "="*60)
        logger.info("CONCATENATING FILES")
        logger.info("="*60)
        
        self.dataframes = []
        
        for filepath in tsv_files:
            df = self.load_and_tag_file(filepath)
            
            if df is None:
                continue
            
            # Validate schema
            if not self.validate_schema(df, filepath):
                logger.warning(f"Skipping {filepath.name} due to schema validation failure")
                continue
            
            # Standardize column names
            df = self.standardize_column_names(df)
            
            self.dataframes.append(df)
        
        if not self.dataframes:
            raise ValueError("No valid dataframes to concatenate")
        
        # Concatenate all dataframes
        logger.info("\nCombining all dataframes...")
        self.combined_df = pd.concat(self.dataframes, ignore_index=True)
        
        logger.info(f"✓ Combined {len(self.dataframes)} files into {len(self.combined_df)} total rows")
        
        return self.combined_df
    
    def add_metadata(self, df):
        """
        Add additional metadata columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Combined dataframe
        
        Returns:
        --------
        pd.DataFrame : Dataframe with additional metadata
        """
        logger.info("\nAdding metadata...")
        
        # Add row ID
        df['Row_ID'] = range(1, len(df) + 1)
        
        # Add processing timestamp
        from datetime import datetime
        df['Processing_Date'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info("✓ Metadata added")
        
        return df
    
    def quality_checks(self, df):
        """
        Perform quality checks on combined data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Combined dataframe
        """
        logger.info("\n" + "="*60)
        logger.info("QUALITY CHECKS")
        logger.info("="*60)
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning("\nMissing values detected:")
            for col, count in missing[missing > 0].items():
                logger.warning(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
        else:
            logger.info("✓ No missing values")
        
        # Check for duplicates
        algo_col = next((col for col in df.columns if 'algorithm' in col.lower()), None)
        go_col = next((col for col in df.columns if col.lower().startswith('go') and 'level' not in col.lower()), None)
        
        if algo_col and go_col:
            duplicates = df.duplicated(subset=[algo_col, go_col, 'GO_Level'], keep=False)
            dup_count = duplicates.sum()
            if dup_count > 0:
                logger.warning(f"\n⚠ Found {dup_count} duplicate rows (same algorithm + GO term + level)")
                logger.info("  Keeping first occurrence of each duplicate")
                df = df[~df.duplicated(subset=[algo_col, go_col, 'GO_Level'], keep='first')]
            else:
                logger.info("✓ No duplicates found")
        
        # Summary statistics
        logger.info("\n" + "-"*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("-"*60)
        
        if algo_col:
            logger.info(f"\nAlgorithms: {df[algo_col].nunique()}")
            for algo in sorted(df[algo_col].unique()):
                count = (df[algo_col] == algo).sum()
                logger.info(f"  {algo}: {count} rows")
        
        if go_col:
            logger.info(f"\nGO Terms: {df[go_col].nunique()}")
            logger.info(f"  Terms: {sorted(df[go_col].unique())}")
        
        if 'GO_Level' in df.columns:
            logger.info(f"\nGO Levels: {df['GO_Level'].nunique()}")
            for level in sorted(df['GO_Level'].unique()):
                count = (df['GO_Level'] == level).sum()
                logger.info(f"  {level}: {count} rows")
        
        return df
    
    def save_combined_data(self, df):
        """
        Save the combined dataframe to file
        
        Parameters:
        -----------
        df : pd.DataFrame
            Combined dataframe to save
        """
        logger.info("\n" + "="*60)
        logger.info("SAVING COMBINED DATA")
        logger.info("="*60)
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as TSV
        df.to_csv(self.output_file, sep='\t', index=False)
        logger.info(f"✓ Saved to: {self.output_file.absolute()}")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {len(df.columns)}")
        
        # Also save as CSV for broader compatibility
        csv_file = self.output_file.with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"✓ Also saved as CSV: {csv_file.absolute()}")
    
    def run(self, file_pattern='*.tsv'):
        """
        Run the complete preprocessing pipeline
        
        Parameters:
        -----------
        file_pattern : str
            Glob pattern to match input files
        
        Returns:
        --------
        pd.DataFrame : Combined and processed dataframe
        """
        logger.info("\n" + "="*70)
        logger.info("GO BENCHMARK DATA PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Find files
        tsv_files = self.find_tsv_files(file_pattern)
        
        if not tsv_files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found in {self.input_dir}")
        
        # Step 2: Load and concatenate
        combined_df = self.concatenate_files(tsv_files)
        
        # Step 3: Add metadata
        combined_df = self.add_metadata(combined_df)
        
        # Step 4: Quality checks
        combined_df = self.quality_checks(combined_df)
        
        # Step 5: Save
        self.save_combined_data(combined_df)
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info("="*70)
        
        self.combined_df = combined_df
        return combined_df


def main():
    """
    Command-line interface for preprocessing
    """
    parser = argparse.ArgumentParser(
        description='Preprocess and concatenate GO term benchmark data from multiple DAG levels'
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing TSV files from different GO levels'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='combined_benchmark_data.tsv',
        help='Output file path (default: combined_benchmark_data.tsv)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.tsv',
        help='File pattern to match (default: *.tsv)'
    )
    
    args = parser.parse_args()
    
    # Run preprocessor
    preprocessor = GODataPreprocessor(args.input_dir, args.output)
    preprocessor.run(file_pattern=args.pattern)


if __name__ == "__main__":
    main()
