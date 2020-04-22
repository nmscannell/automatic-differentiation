from autodiff import evaluate, differentiate, add, multiply
import unittest


class TestAutodiff(unittest.TestCase):

    def test_add_const_evaluate(self):
        x = add(3, 4)
        self.assertIsInstance(x, add)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)
        self.assertEqual(x.evaluate({}), 7)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)

    def test_add_const_diff(self):
        x = add(3, 4)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)
        self.assertEqual(1, x.local_grad1)
        self.assertEqual(1, x.local_grad2)
        self.assertEqual(x.diff({}), 0)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)
        self.assertEqual(0, x.local_grad1)
        self.assertEqual(0, x.local_grad2)

    def test_add_var_evaluate(self):
        x = add(3, 'x')
        self.assertEqual(3, x.in1)
        self.assertEqual('x', x.in2)
        self.assertEqual(x.evaluate({'x': 2}), 5)
        self.assertEqual(3, x.in1)
        self.assertEqual(2, x.in2)

    def test_add_var_diff(self):
        x = add(3, 'x')
        self.assertEqual(3, x.in1)
        self.assertEqual('x', x.in2)
        self.assertEqual(1, x.local_grad1)
        self.assertEqual(1, x.local_grad2)
        self.assertEqual(x.diff({'x': 2}), 1)
        self.assertEqual(3, x.in1)
        self.assertEqual(2, x.in2)
        self.assertEqual(0, x.local_grad1)
        self.assertEqual(1, x.local_grad2)

    def test_multiply_const_evaluate(self):
        x = multiply(3, 4)
        self.assertIsInstance(x, multiply)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)
        self.assertEqual(x.evaluate({}), 12)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)

    def test_multiply_const_diff(self):
        x = multiply(3, 4)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)
        self.assertEqual(4, x.local_grad1)
        self.assertEqual(3, x.local_grad2)
        self.assertEqual(x.diff({}), 0)
        self.assertEqual(3, x.in1)
        self.assertEqual(4, x.in2)
        self.assertEqual(0, x.local_grad1)
        self.assertEqual(0, x.local_grad2)

    def test_multiply_var_evaluate(self):
        x = multiply(3, 'x')
        self.assertIsInstance(x, multiply)
        self.assertEqual(3, x.in1)
        self.assertNotEqual(3, x.in2)
        self.assertEqual('x', x.in2)
        self.assertNotEqual('x', x.in1)
        self.assertEqual(x.evaluate({'x': 2}), 6)

    def test_multiply_var_diff(self):
        x = multiply(3, 'x')
        self.assertEqual(3, x.in1)
        self.assertEqual('x', x.in2)
        self.assertEqual('x', x.local_grad1)
        self.assertEqual(3, x.local_grad2)
        self.assertEqual(x.diff({'x': 2}), 3)
        self.assertEqual(3, x.in1)
        self.assertEqual(2, x.in2)
        self.assertEqual(0, x.local_grad1)
        self.assertEqual(1, x.local_grad2)

    def test_evaluate_no_ops(self):
        self.assertEqual(evaluate(3, {'x': 2}), 3)
        self.assertEqual(evaluate('x', {'x': 2}), 2)
        self.assertEqual(evaluate('y', {'y': 2}), 2)

    def test_evaluate_simple_add(self):
        self.assertEqual(evaluate(add(3, 'x'), {'x': 2}), 5)
        self.assertEqual(evaluate(add(3, 3), {}), 6)
        self.assertEqual(evaluate(add('x', 'y'), {'x': 1, 'y': 2}), 3)

    def test_evaluate_compound_add(self):
        self.assertEqual(evaluate(add(3, add('x', 4)), {'x': 2}), 9)
        self.assertEqual(evaluate(add('x', add('y', 3)), {'x': 2, 'y': 5}), 10)

    def test_evaluate_simple_multiply(self):
        self.assertEqual(evaluate(multiply(3, 3), {}), 9)
        self.assertEqual(evaluate(multiply(3, 'x'), {'x': 2, 'y': 5}), 6)
        self.assertEqual(evaluate(multiply('x', 'y'), {'x': 2, 'y': 5}), 10)
        self.assertEqual(evaluate(multiply('x', 'x'), {'x': 2, 'y': 5}), 4)

    def test_evaluate_compound_multiply(self):
        self.assertEqual(evaluate(multiply('x', multiply(3, 'y')), {'x': 2, 'y': 1}), 6)
        self.assertEqual(evaluate(multiply(3, multiply('x', multiply('x', 'x'))), {'x': 2}), 24)

    def test_evaluate_compound_ops(self):
        self.assertEqual(evaluate(add(3, multiply('x', 'x')), {'x': 2, 'y': 5}), 7)
        self.assertEqual(evaluate(add(3, add(multiply(2, 'x'), multiply('x', multiply('x', 'x')))), {'x': 2}), 15)

    def test_differentiate_no_ops(self):
        self.assertEqual(differentiate(3, {'x': 2}), 0)
        self.assertEqual(differentiate('x', {'x': 2}), 1)
        self.assertEqual(differentiate('y', {'y': 2}), 0)

    def test_differentiate_simple_add(self):
        self.assertEqual(differentiate(add(3, 'x'), {'x': 2}), 1)
        self.assertEqual(differentiate(add(3, 3), {}), 0)
        self.assertEqual(differentiate(add('x', 'y'), {'x': 1, 'y': 2}), 1)

    def test_differentiate_compound_add(self):
        self.assertEqual(differentiate(add(3, add('x', 4)), {'x': 2}), 1)
        self.assertEqual(differentiate(add('x', add('y', 3)), {'x': 2, 'y': 5}), 1)

    def test_differentiate_simple_multiply(self):
        self.assertEqual(differentiate(multiply(3, 3), {}), 0)
        self.assertEqual(differentiate(multiply(3, 'x'), {'x': 2, 'y': 5}), 3)
        self.assertEqual(differentiate(multiply('x', 'y'), {'x': 2, 'y': 5}), 5)
        self.assertEqual(differentiate(multiply('x', 'x'), {'x': 2, 'y': 5}), 4)

    def test_differentiate_compound_multiply(self):
        self.assertEqual(differentiate(multiply('x', multiply(3, 'y')), {'x': 2, 'y': 1}), 3)
        self.assertEqual(differentiate(multiply(3, multiply('x', multiply('x', 'x'))), {'x': 2}), 36)

    def test_differentiate_compount_ops(self):
        self.assertEqual(differentiate(add(3, multiply('x', 'x')), {'x': 2, 'y': 5}), 4)
        self.assertEqual(differentiate(add(3, add(multiply(2, 'x'), multiply('x', multiply('x', 'x')))), {'x': 2}), 14)


if __name__ == '__main__':
    unittest.main()
