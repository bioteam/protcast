import sys
import argparse
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.preprocessing.parse_gaf import parse_gaf  # noqa: E402


"""test_gaf_parser.py
Checks that a GAF file is parsed correctly
"""
parser = argparse.ArgumentParser(
    description="Checks that a GAF file is parsed correctly"
)
parser.add_argument("-i", "--input", default="data/goa_uniprot_mini.gaf")
args = parser.parse_args()

annotations = parse_gaf(args.input)

assert len(annotations) == 995
assert annotations[0]["DB_Object_ID"] == "A0A1Z4V764"
assert annotations[990]["DB_Object_ID"] == "A0A6N9GJR9"
assert annotations[994]["DB_Object_ID"] == "M5BGM1"
assert annotations[994]["GO_ID"] == "GO:0046872"
