import os
from pathlib import Path
import pytest
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot
import shutil


def test_goatools_plots(tmp_path):
    """Create small GO subdag plots using the test data OBO file.
    This is a smoke test; it will be skipped if the OBO fixture is missing.
    """
    gofile = Path(__file__).parent / "data" / "go-2023-11-15.obo"
    if not gofile.exists():
        pytest.skip("go OBO fixture not present: skipping goatools plot test")

    godag = GODag(str(gofile))

    # GoSubDag contains only the terms specified on instantiation
    gosubdag = GoSubDag("GO:0015645", godag)
    goploter = GoSubDagPlot(gosubdag)

    # If graphviz 'dot' is not installed, skip plotting tests
    if shutil.which("dot") is None:
        pytest.skip("GraphViz 'dot' binary not found in PATH; skipping plot creation")

    # Plot ancestors and descendants into a tmp directory
    anc_file = tmp_path / "ancestors-test.png"
    desc_file = tmp_path / "descendants-test.png"
    goploter.plt_dag(str(anc_file))

    descendants = gosubdag.rcntobj.go2descendants["GO:0015645"]
    assert len(descendants) == 15

    # Initialize DAG with descendants and plot
    gosubdag = GoSubDag(descendants, godag)
    goploter = GoSubDagPlot(gosubdag)
    goploter.plt_dag(str(desc_file))

    assert anc_file.exists()
    assert desc_file.exists()
