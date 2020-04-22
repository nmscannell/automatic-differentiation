from autodiff import evaluate, differentiate, add, multiply
import unittest


class TestAutodiff(unittest.TestCase):

    def test_add(self):
        x = add(3, 4)
        self.assertIsInstance(x, add)
        self.assertEqual(3, x.in1)
        self.assertNotEqual(3, x.in2)
        self.assertEqual(4, x.in2)
        self.assertNotEqual(4, x.in1)
        self.assertEqual(x.evaluate({}), 7)
        self.assertEqual(x.diff({}), 0)

    def test_evaluate(self):
        self.assertEqual(evaluate(3, {'x': 2}), 3)
        self.assertEqual(evaluate('x', {'x': 2}), 2)
        self.assertEqual(evaluate('y', {'y': 2}), 2)
        self.assertEqual(evaluate(add(3, 'x'), {'x': 2}), 5)
        self.assertEqual(evaluate(add(3, 3), {}), 6)
        self.assertEqual(evaluate(add('x', 'y'), {'x': 1, 'y': 2}), 3)
        self.assertEqual(evaluate(add(3, add('x', 4)), {'x': 2}), 9)
        self.assertEqual(evaluate(multiply(3, 3), {}), 9)
        self.assertEqual(evaluate(multiply(3, 'x'), {'x': 2, 'y': 5}), 6)
        self.assertEqual(evaluate(multiply('x', 'y'), {'x': 2, 'y': 5}), 10)
        self.assertEqual(evaluate(multiply('x', 'x'), {'x': 2, 'y': 5}), 4)
        self.assertEqual(evaluate(add(3, multiply('x', 'x')), {'x': 2, 'y': 5}), 7)
        self.assertEqual(evaluate(add(3, add(multiply(2, 'x'), multiply('x', multiply('x', 'x')))), {'x': 2}), 15)

    def test_differentiate(self):
        self.assertEqual(differentiate(3, {'x': 2}), 0)
        self.assertEqual(differentiate('x', {'x': 2}), 1)
        self.assertEqual(differentiate('y', {'y': 2}), 0)
        self.assertEqual(differentiate(add(3, 'x'), {'x': 2}), 1)
        self.assertEqual(differentiate(add(3, 3), {}), 0)
        self.assertEqual(differentiate(add('x', 'y'), {'x': 1, 'y': 2}), 1)
        self.assertEqual(differentiate(add(3, add('x', 4)), {'x': 2}), 1)
        self.assertEqual(differentiate(multiply(3, 3), {}), 0)
        self.assertEqual(differentiate(multiply(3, 'x'), {'x': 2, 'y': 5}), 3)
        self.assertEqual(differentiate(multiply('x', 'y'), {'x': 2, 'y': 5}), 5)
        self.assertEqual(differentiate(multiply('x', 'x'), {'x': 2, 'y': 5}), 4)
        self.assertEqual(differentiate(add(3, multiply('x', 'x')), {'x': 2, 'y': 5}), 4)
        self.assertEqual(differentiate(add(3, add(multiply(2, 'x'), multiply('x', multiply('x', 'x')))), {'x': 2}), 14)
