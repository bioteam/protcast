# GO Benchmark Pipeline - Quick Reference Card

## 🚀 Installation & Setup

```bash
# Option 1: Automated setup
chmod +x setup.sh
./setup.sh

# Option 2: Manual setup
conda env create -f environment.yml
conda activate go_benchmark_analysis
make setup
```

## 📂 Prepare Your Data

```bash
# Directory structure
data/raw/
├── level_3.tsv    # GO level 3 results
├── level_4.tsv    # GO level 4 results
└── level_5.tsv    # GO level 5 results
```

Required columns (names are flexible):
- Algorithm, GO_Term, F1_Score, Sensitivity, Specificity

Optional columns:
- Vector_Length, Elapsed_Time, TP, TN, FP, FN

## ⚡ Run Pipeline

```bash
# Complete pipeline (one command)
make all

# Or step by step
make preprocess    # Concatenate TSV files
make analyze       # Run analysis

# Test with demo data
make test
```

## 📊 Check Results

```bash
results/
├── figures/
│   ├── efficiency_frontier.png      # F1 vs cost
│   ├── algorithm_comparison.png     # Boxplots
│   └── correlation_heatmap.png      # Error correlation
├── reports/
│   ├── analysis_summary.txt         # Key findings
│   └── benchmark_summary_report.csv # Statistics
└── data/
    └── enhanced_benchmark_data.csv  # With calculated metrics
```

## 🔧 Configuration

Edit `config.yaml`:
```yaml
paths:
  raw_data_dir: "data/raw"
  file_pattern: "*.tsv"
  
analysis:
  calculate_balanced_accuracy: true
  calculate_mcc: true
  
output:
  figure_format: "png"
  figure_dpi: 300
```

## 🐍 Python API

```python
# Preprocess
from preprocess import GODataPreprocessor
preprocessor = GODataPreprocessor('data/raw', 'combined.tsv')
preprocessor.run()

# Analyze
from run_pipeline import AnalysisPipeline
pipeline = AnalysisPipeline('combined.tsv', 'results')
pipeline.run()
```

## 🔍 Make Commands

```bash
make all         # Full pipeline
make setup       # Create directories
make preprocess  # Concatenate files
make analyze     # Run analysis
make test        # Demo with sample data
make clean       # Remove results
make help        # Show all commands
```

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| No files found | Check `data/raw/` and file pattern |
| Column not found | Names matched case-insensitively |
| Missing dependencies | `pip install -r requirements.txt` |
| Memory error | Reduce figure DPI in config.yaml |

## 📝 For Your Paper

Key metrics from `results/reports/analysis_summary.txt`:

1. **Best baseline algorithm** and mean F1 score
2. **Efficiency metrics** (F1 per dimension, F1 per second)
3. **Head vs tail performance** gap
4. **Algorithm correlation** (complementary signals)

Example text:
```
"The best performing baseline was CKSAAP with F1 = 0.82 ± 0.06 
across 5 GO terms. Performance stratification showed F1_head = 0.89 
vs F1_tail = 0.65, indicating significant room for improvement 
on rare/specific terms."
```

## 🔄 Update Pipeline

```bash
git pull                    # Get updates
conda env update -f environment.yml  # Update dependencies
make clean && make all     # Re-run
```

## 📚 File Reference

| File | Purpose |
|------|---------|
| `preprocess.py` | Concatenate TSV files |
| `benchmark_analysis.py` | Analysis class |
| `run_pipeline.py` | Pipeline orchestrator |
| `config.yaml` | Configuration |
| `Makefile` | Automation |
| `environment.yml` | Conda environment |
| `requirements.txt` | Pip dependencies |

## ⌨️ Common Workflows

### First Time Setup
```bash
./setup.sh
# Add TSV files to data/raw/
make test
make all
```

### After Adding New Data
```bash
# Add files to data/raw/
make preprocess
make analyze
```

### Re-run Analysis Only
```bash
make analyze
```

### Clean and Start Fresh
```bash
make clean
make all
```

## 💡 Tips

- Keep raw data in `data/raw/` and never edit it
- Use descriptive filenames with level indicators
- Version control `config.yaml` and scripts (not data)
- Check `analysis_summary.txt` for key findings
- Use `make test` to verify before processing real data

## 🆘 Getting Help

```bash
make help                  # See all make commands
python preprocess.py -h    # Preprocessing options
python run_pipeline.py -h  # Pipeline options
```

Read PIPELINE_README.md for detailed documentation.
