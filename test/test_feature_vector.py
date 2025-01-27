import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

from protcast.model.feature_vector import FeatureVector  # noqa: E402

fv = FeatureVector()
algs = fv.get_feature_vector_names("ifeatpro")
assert len(algs) == 21
algs = fv.get_feature_vector_names("iFeatureOmega")
assert len(algs) == 49
