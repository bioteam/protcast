import re
import sys
import time
import json
import argparse
from collections import defaultdict
from Bio import SeqIO
from pathlib import Path
import inspect

# Add project root to sys.path for protcast imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.config.model_config import ConfigManager  # noqa: E402

parser = argparse.ArgumentParser(
    description="Test Multi-Classifier with configurable parameters"
)
parser.add_argument(
    "-s", "--seq_file", required=True, help="Path to Fasta file"
)
parser.add_argument(
    "--use_tensorboard", action="store_true", help="Use TensorBoard logging"
)
parser.add_argument(
    "--use_mlflow", action="store_true", help="Use MLFlow logging"
)
parser.add_argument(
    "-a", "--algorithm", default="CTriad", help="Feature vector algorithm"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Verbose output"
)
parser.add_argument(
    "--config_override", type=str, help="JSON string to override config values"
)
parser.add_argument(
    "--config_path", type=str, help="Path to custom config file"
)
args = parser.parse_args()

start = time.time()

if args.verbose:
    print("MultiClassifier defined in:", inspect.getfile(MultiClassifier))

# Collect proteins from FASTA
proteins = defaultdict(dict)

for seq in SeqIO.parse(args.seq_file, "fasta"):
    match = re.search(r"GO:\d+", seq.description)
    if match:
        go_id = match.group(0)
        proteins[go_id][seq.id] = str(seq.seq)
    elif args.verbose:
        print(f"Warning: No GO ID found in sequence {seq.id}")

if not proteins:
    print("Error: No sequences with GO IDs found in the input file")
    sys.exit(1)

if args.verbose:
    print(f"Number of GO ids: {len(proteins.keys())}")
    print(f"GO IDs found: {list(proteins.keys())}")
    total_proteins = sum(len(inner_dict) for inner_dict in proteins.values())
    print(f"Number of proteins: {total_proteins}")


if args.config_path is not None:
    config = ConfigManager.load_config(args.config_path)
    if args.verbose:
        print(f"Using config file: {args.config_path}")
else:
    config = ConfigManager.load_config()

# Use config overrides if provided
if args.config_override:
    try:
        config_override = json.loads(args.config_override)
        if args.verbose:
            print(f"Config overrides: {json.dumps(config_override, indent=2)}")
        config.update(config_override)
    except json.JSONDecodeError as e:
        print(f"Error parsing config_override JSON: {e}")
        sys.exit(1)


classifier = MultiClassifier(
    args.algorithm,
    args.verbose,
    proteins,
    config,
    use_mlflow=args.use_mlflow,
    use_tensorboard=args.use_tensorboard,
)

if args.verbose:
    print(f"\nInitializing MultiClassifier with algorithm: {args.algorithm}")

try:
    classifier.run()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    if args.verbose:
        import traceback

        traceback.print_exc()
    sys.exit(1)

end = time.time()
elapsed_time = round(end - start)

if args.verbose:
    print("\nTraining Summary:")
    print(f"\tAlgorithm: {args.algorithm}")
    print(f"\tTotal elapsed time: {elapsed_time}s")
    print(
        f"\tFinal validation loss: {getattr(classifier, 'final_val_loss', 'N/A')}"
    )
    if hasattr(classifier, "model"):
        print(f"\tModel parameters: {classifier.model.count_params():,}")

print(f"Elapsed {args.algorithm} time: {elapsed_time}s")
