import tfmpl
import numpy as np

def test_confusion_matrix():
    cm = tfmpl.plots.confusion_matrix.from_labels_and_predictions([1, 2, 4], [2, 2, 4], num_classes=5)
    exp = np.zeros((5,5), dtype=int)
    exp[1,2] = exp[2,2] = exp[4,4] = 1
    np.testing.assert_allclose(cm, exp)
