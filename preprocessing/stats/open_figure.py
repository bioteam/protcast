import argparse
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.use("Qt5Agg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", help ="Path to figure")

    args = parser.parse_args()

    figx = pickle.load(open(args.figure, 'rb'))
    plt.show()

