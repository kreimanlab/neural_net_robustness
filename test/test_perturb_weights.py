import copy
import functools
import unittest

from perturb_weights import __perturb_all as perturb, __draw as draw
from weights import load_weights, proportion_different


class PerturbAlexnetWeightsTests(unittest.TestCase):
    _weights = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = None

    @classmethod
    def setUpClass(cls):
        cls._weights = load_weights('alexnet')

    def setUp(self):
        self.weights = copy.deepcopy(PerturbAlexnetWeightsTests._weights)

    def test_draw_conv_1(self):
        self._test_draw_layer('conv_1')

    def test_draw_conv_2(self):
        self._test_draw_layer('conv_2')

    def test_draw_conv_3(self):
        self._test_draw_layer('conv_3')

    def test_draw_conv_4(self):
        self._test_draw_layer('conv_4')

    def test_draw_conv_5(self):
        self._test_draw_layer('conv_5')

    def _test_draw_layer(self, layer, proportion=0.5):
        perturb(self.weights, layer, functools.partial(draw, proportion=proportion))
        # perturbed layers did change
        affected_layers = [w for w in PerturbAlexnetWeightsTests._weights.keys() if w.startswith(layer)]
        self._compare_layers(affected_layers, PerturbAlexnetWeightsTests._weights, self.weights,
                             self._assert_affected)
        # no other layers changed
        unaffected_layers = PerturbAlexnetWeightsTests._weights.keys() - affected_layers
        self._compare_layers(unaffected_layers, PerturbAlexnetWeightsTests._weights, self.weights,
                             self._assert_unaffected)

    def _assert_unaffected(self, base_weights, compare_weights, layer):
        self.assertEqual(0, proportion_different(base_weights, compare_weights),
                         "weights in layer %s differ" % layer)

    def _assert_affected(self, base_weights, compare_weights, layer):
        self.assertGreater(proportion_different(base_weights, compare_weights), 0,
                           "weights in layer %s do not differ" % layer)

    def _compare_layers(self, layers, base_weights, compare_weights, compare_fnc):
        for layer in layers:
            if not base_weights[layer]:
                self.assertEqual(0, len(compare_weights[layer]))
            else:
                compare_fnc(base_weights[layer], compare_weights[layer], layer)


if __name__ == '__main__':
    unittest.main()
