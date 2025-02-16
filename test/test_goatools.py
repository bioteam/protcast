import argparse
import os
from pathlib import Path
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot


"""test_goatools.py
Checks goatools results and creates plots
"""
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gofile", default="data/go.obo")
parser.add_argument("-k", "--keep", default=False, action="store_true")
args = parser.parse_args()

godag = GODag(args.gofile)

# GoSubDag contains only the terms specified on instantiation
gosubdag = GoSubDag("GO:0015645", godag)
goploter = GoSubDagPlot(gosubdag)
# Plot ancestors
goploter.plt_dag("ancestors-test.png")

descendants = gosubdag.rcntobj.go2descendants["GO:0015645"]
assert len(descendants) == 15
# Initialize DAG with descendants
gosubdag = GoSubDag(descendants, godag)
goploter = GoSubDagPlot(gosubdag)
# Plot ancestors and descendants
goploter.plt_dag("descendants-test.png")

assert os.path.isfile(Path("descendants-test.png"))
assert os.path.isfile(Path("ancestors-test.png"))

if not args.keep:
    os.unlink(Path("ancestors-test.png"))
    os.unlink(Path("descendants-test.png"))
