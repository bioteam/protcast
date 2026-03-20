# GO Benchmark Analysis Pipeline

**Automated, reproducible pipeline for analyzing protein function prediction benchmarks**

## 🎯 What This Pipeline Does

This is a complete, automated pipeline for rigorous benchmarking of protein GO term prediction methods. It handles:

1. **Data Preprocessing**: Concatenates TSV files from multiple GO DAG levels
2. **Metric Calculation**: Balanced accuracy, MCC, efficiency metrics
3. **Statistical Analysis**: Macro-averaging, term stratification, error correlation
4. **Visualization**: Efficiency frontiers, performance comparisons, correlation heatmaps
5. **Reporting**: Automated summary reports with key findings

## 🚀 Quick Start

### Option 1: Using Make (Recommended)

```bash
# 1. Setup environment
make setup

# 2. Place your TSV files in data/raw/
# Each file should be from a different GO DAG level, e.g.:
#   - level_3.tsv
#   - level_4.tsv
#   - level_5.tsv

# 3. Run the complete pipeline
make all

# Results will be in the results/ directory
```

### Option 2: Manual Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt
# OR with conda:
conda env create -f environment.yml
conda activate go_benchmark_analysis

# 2. Create directory structure
mkdir -p data/raw data/processed results

# 3. Preprocess data
python preprocess.py data/raw \
    --output data/processed/combined_benchmark_data.tsv \
    --pattern "*.tsv"

# 4. Run analysis
python run_pipeline.py \
    --input data/processed/combined_benchmark_data.tsv \
    --output results
```

### Option 3: Test with Demo Data

```bash
# Run with synthetic data to verify installation
make test
```

## 📁 Directory Structure

```
go-benchmark-pipeline/
├── data/
│   ├── raw/                    # Place your TSV files here
│   │   ├── level_3.tsv
│   │   ├── level_4.tsv
│   │   └── level_5.tsv
│   └── processed/              # Combined data (auto-generated)
│       └── combined_benchmark_data.tsv
├── results/                     # All outputs go here
│   ├── figures/                # Plots and visualizations
│   ├── reports/                # Summary statistics
│   └── data/                   # Enhanced data with metrics
├── preprocess.py               # Data preprocessing script
├── benchmark_analysis.py       # Core analysis class
├── run_pipeline.py            # Main pipeline orchestrator
├── config.yaml                # Configuration file
├── Makefile                   # Automated workflow
└── requirements.txt           # Python dependencies
```

## 📊 Input Data Format

Your TSV files should contain columns like:

| Column | Required | Description |
|--------|----------|-------------|
| Algorithm | Yes | Feature extraction method (aac, cksaap, etc.) |
| GO_Term | Yes | GO term ID (e.g., GO:0003678) |
| F1_Score | Yes | F1 score for the prediction |
| Sensitivity | Yes | True positive rate (recall) |
| Specificity | Yes | True negative rate |
| Vector_Length | Optional | Dimensionality of feature vector |
| Elapsed_Time | Optional | Computation time (ms) |
| TP, TN, FP, FN | Optional | Confusion matrix values (for MCC) |

**Column names are flexible** - the pipeline will find them using case-insensitive matching.

### Example Input Files

**data/raw/level_3.tsv:**
```tsv
Algorithm	GO_Term	Sensitivity	Specificity	F1_Score	Vector_Length	Elapsed_Time
aac	GO:0003678	0.85	0.92	0.88	20	15.2
cksaap	GO:0003678	0.90	0.95	0.92	2400	39000
```

**data/raw/level_4.tsv:**
```tsv
Algorithm	GO_Term	Sensitivity	Specificity	F1_Score	Vector_Length	Elapsed_Time
aac	GO:0004721	0.72	0.88	0.79	20	15.1
cksaap	GO:0004721	0.82	0.94	0.87	2400	38500
```

## 🎛️ Configuration

Edit `config.yaml` to customize:

- Input/output paths
- File patterns
- Which metrics to calculate
- Visualization options
- Logging level

## 📈 Output Files

The pipeline generates:

### Figures (results/figures/)
- `efficiency_frontier.png` - F1 vs computational cost
- `algorithm_comparison.png` - Performance distribution boxplots
- `correlation_heatmap.png` - Algorithm error correlation matrix

### Reports (results/reports/)
- `benchmark_summary_report.csv` - Statistical summary by algorithm
- `analysis_summary.txt` - Human-readable text summary

### Data (results/data/)
- `enhanced_benchmark_data.csv` - Original data + calculated metrics

## 🔄 Pipeline Workflow

```
┌─────────────────────┐
│  Raw TSV Files      │ (level_3.tsv, level_4.tsv, ...)
│  (different GO      │
│   DAG levels)       │
└──────────┬──────────┘
           │
           ├── preprocess.py
           │   • Concatenate files
           │   • Add GO_Level metadata
           │   • Standardize column names
           │   • Validate & check quality
           │
           ▼
┌─────────────────────┐
│  Combined Data      │ (combined_benchmark_data.tsv)
└──────────┬──────────┘
           │
           ├── run_pipeline.py
           │   • Load data
           │   • Calculate metrics
           │   • Aggregate statistics
           │   • Generate visualizations
           │   • Export results
           │
           ▼
┌─────────────────────┐
│  Results            │
│  • Figures          │
│  • Reports          │
│  • Enhanced Data    │
└─────────────────────┘
```

## 🔧 Make Targets

```bash
make all         # Run complete pipeline
make setup       # Create directories, check dependencies
make preprocess  # Concatenate TSV files
make analyze     # Run analysis on preprocessed data
make test        # Run with demo data
make clean       # Remove generated files
make help        # Show available commands
```

## 🐍 Using as a Python Library

```python
from preprocess import GODataPreprocessor
from run_pipeline import AnalysisPipeline

# Preprocess data
preprocessor = GODataPreprocessor('data/raw', 'data/processed/combined.tsv')
preprocessor.run()

# Run analysis
pipeline = AnalysisPipeline('data/processed/combined.tsv', 'results')
pipeline.run()
```

## 🧪 Environment Management

### Using Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate go_benchmark_analysis

# Deactivate when done
conda deactivate
```

### Using venv

```bash
# Create environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate
# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## 📝 Configuration Options

### Input/Output Paths

```yaml
paths:
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  results_dir: "results"
  file_pattern: "*.tsv"
```

### Analysis Options

```yaml
analysis:
  calculate_balanced_accuracy: true
  calculate_mcc: true
  calculate_efficiency_metrics: true
  aggregate_by_algorithm: true
  analyze_term_stratification: true
  calculate_error_correlation: true
```

### Output Options

```yaml
output:
  save_enhanced_data: true
  generate_summary: true
  figure_format: "png"
  figure_dpi: 300
```

## 🔍 Troubleshooting

### "No files found" error

- Check that your TSV files are in `data/raw/`
- Verify file pattern in config.yaml matches your filenames
- Files from different GO levels should have level indicators in filenames (e.g., `level_3.tsv`)

### "Column not found" error

- Column names are matched case-insensitively
- Check the preprocessing output to see detected column names
- Supported variations: algo/algorithm, go/goterm, f1/f1_score, etc.

### "Missing dependencies" error

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or with conda
conda env remove -n go_benchmark_analysis
conda env create -f environment.yml
```

### Memory issues with large datasets

- Process files in batches
- Reduce figure DPI in config.yaml
- Use sampling for initial exploration

## 🎓 For Your Paper

After running the pipeline, your `results/reports/analysis_summary.txt` contains key findings to report:

### Example Results Section

```
"We established baseline performance using traditional feature extraction 
methods across 5 GO terms spanning levels 3-5 of the molecular function 
ontology. The best performing method was CKSAAP (F1 = 0.8234 ± 0.0567), 
which we use as the hard baseline for comparison. Performance showed 
significant stratification between head terms (F1 = 0.89) and tail terms 
(F1 = 0.65), indicating room for improvement on rare/specific annotations."
```

### Key Metrics to Report

1. **Best baseline name and performance**
2. **Efficiency metrics** (F1/dimension, F1/second)
3. **Head vs tail performance gap**
4. **Algorithm error correlation**
5. **Number of terms, algorithms, and GO levels**

## 🔐 Reproducibility

The pipeline ensures reproducibility through:

1. **Version Control**: Track `config.yaml` and scripts in git
2. **Environment Files**: `environment.yml` pins exact versions
3. **Automated Workflow**: `make all` runs identically every time
4. **Logging**: All steps logged with timestamps
5. **Configuration**: All parameters in `config.yaml`

### Recommended Workflow

```bash
# 1. Version control your setup
git init
git add config.yaml environment.yml Makefile
git commit -m "Initial pipeline setup"

# 2. Add your data (or .gitignore it)
# Don't commit large data files to git
echo "data/" >> .gitignore

# 3. Run pipeline
make all

# 4. Commit results metadata (not large files)
git add results/reports/analysis_summary.txt
git commit -m "Analysis results for experiment X"
```

## 📚 What This Analysis Is Called

In academic literature, this work is referred to as:

- **Benchmarking Analysis** or **Comparative Model Evaluation**
- **Performance-Cost Tradeoff Analysis** (efficiency frontier)
- **Imbalanced Classification Evaluation** (MCC, balanced accuracy)
- **Hierarchical Classification Analysis** (GO term-specific metrics)
- **Baseline Comparison Study** or **Ablation Study**

## 📖 Citation

If you use this pipeline, consider citing the methodological papers:

```bibtex
@article{radivojac2013cafa,
  title={A large-scale evaluation of computational protein function prediction},
  author={Radivojac, Predrag and Clark, Wyatt T and others},
  journal={Nature methods},
  year={2013}
}

@article{matthews1975mcc,
  title={Comparison of the predicted and observed secondary structure of T4 phage lysozyme},
  author={Matthews, Brian W},
  journal={Biochimica et Biophysica Acta},
  year={1975}
}
```

## 🤝 Contributing

To extend the pipeline:

1. Add new analysis functions to `benchmark_analysis.py`
2. Update `run_pipeline.py` to call them
3. Add configuration options to `config.yaml`
4. Update `Makefile` if needed

## 📄 License

MIT License - Free to use and modify for research purposes.

## ❓ Questions?

For issues specific to:
- **Pipeline setup**: Check environment.yml and requirements.txt
- **Data format**: See "Input Data Format" section
- **GO semantics**: Consult UniProt-GOA and GO documentation
- **CAFA standards**: See CAFA assessment documentation

## 🎉 You're Ready!

```bash
# Start here:
make test    # Verify everything works

# Then:
# 1. Add your TSV files to data/raw/
# 2. Run: make all
# 3. Check results/ directory
# 4. Use findings in your paper!
```
