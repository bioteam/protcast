import argparse
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot


if __name__ == "__main__":
    """test_goatools.py
    Checks goatools results and creates plots
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="data/go.obo")
    args = parser.parse_args()

    godag = GODag(args.input)
    
    # GoSubDag contains only the terms specified on instantiation
    gosubdag = GoSubDag("GO:0015645", godag)
    goploter = GoSubDagPlot(gosubdag)
    # Plots ancestors by default
    goploter.plt_dag("ancestors-test.png")

    descendants = gosubdag.rcntobj.go2descendants["GO:0015645"]
    assert len(descendants) == 15
    gosubdag = GoSubDag(descendants, godag)
    goploter = GoSubDagPlot(gosubdag)
    # Plots descendants
    goploter.plt_dag("descendants-test.png")
