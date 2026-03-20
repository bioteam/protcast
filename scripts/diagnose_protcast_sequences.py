#!/usr/bin/env python3
"""
Diagnostic script to investigate sequence lengths in ProtCastDataset
"""

from protcast.preprocessing.protcast_dataset import ProtCastDataset

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose ProtCastDataset sequences")
    parser.add_argument("-p", "--protcast_dataset", required=True, help="Path to ProtCastDataset")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.protcast_dataset}...")
    dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)
    
    print(f"Total proteins in dataset: {len(dataset.proteins)}")
    print()
    
    # Sample a few proteins to examine
    protein_ids = list(dataset.proteins.keys())[:5]
    
    for pid in protein_ids:
        print(f"=== Protein: {pid} ===")
        protein_obj = dataset.proteins[pid]
        print(f"  Type: {type(protein_obj)}")
        print(f"  Object: {protein_obj}")
        
        if hasattr(protein_obj, '__dict__'):
            print(f"  Attributes: {list(protein_obj.__dict__.keys())}")
            for attr, value in protein_obj.__dict__.items():
                if attr == 'sequence':
                    print(f"    {attr}:")
                    print(f"      Type: {type(value)}")
                    if hasattr(value, '__len__'):
                        print(f"      Length: {len(value)}")
                    if isinstance(value, str):
                        print(f"      First 100 chars: {value[:100]}")
                        print(f"      Last 100 chars: {value[-100:]}")
                        # Check if it looks like a valid protein sequence
                        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                        seq_chars = set(value.upper())
                        invalid_chars = seq_chars - valid_aa
                        if invalid_chars:
                            print(f"      WARNING: Contains invalid amino acids: {invalid_chars}")
                else:
                    print(f"    {attr}: {value}")
        print()
    
    # Get statistics on sequence lengths
    print("=== Sequence Length Statistics ===")
    lengths = []
    for pid, prot in list(dataset.proteins.items())[:100]:  # Sample first 100
        if hasattr(prot, 'sequence'):
            seq = prot.sequence
            if isinstance(seq, str):
                lengths.append(len(seq))
    
    if lengths:
        lengths.sort()
        print(f"Sample size: {len(lengths)} proteins")
        print(f"Min length: {min(lengths)}")
        print(f"Max length: {max(lengths)}")
        print(f"Median length: {lengths[len(lengths)//2]}")
        print(f"Mean length: {sum(lengths)/len(lengths):.1f}")
        print(f"Sequences > 10000: {sum(1 for l in lengths if l > 10000)}")
        print(f"Sequences > 5000: {sum(1 for l in lengths if l > 5000)}")
        print(f"Sequences > 1000: {sum(1 for l in lengths if l > 1000)}")

if __name__ == "__main__":
    main()
