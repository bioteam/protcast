import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

from protcast.model.feature_vector import FeatureVector  # noqa: E402

seq = {
    "A12345": "LVTIKIGGQLKEALLDTGADDTVLEDMHLPGKWKPKMIGGIGGFIKVRQYDQILVEICGH"
    + "KAIGTVLVGPTPVNIIGRNLLTQIGCTLNFEMEKEGKISKIGPENPYNTPIFAIKKKDST"
    + "KWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDEDFRKYTA"
    + "FTIPSTNNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYV"
    + "GSDLEIGQHRIKVEELRQHLLRWGLTTPDKKHQKEPPFLWMG"
}

fv = FeatureVector(verbose=True, feature_creator="ifeatpro")
algs = fv.get_feature_vector_names()
assert len(algs) == 21
for alg in algs:
    fv.get_feature_vectors(seq, algorithm=alg)
fv = FeatureVector(verbose=True, feature_creator="iFeatureOmega")
algs = fv.get_feature_vector_names()
assert len(algs) == 49
for alg in algs:
    fv.get_feature_vectors(seq, algorithm=alg)
