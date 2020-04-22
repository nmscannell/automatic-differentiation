"""
Auto-diff for Anyscale
"""


def evaluate(expr, value_map):
    """Evaluate expr for a specific value of x.

    Examples:
        >>> evaluate(multiply(3, 'x'), {'x':2})
        0  # because 3*0 = 0
        >>> evaluate(add(3, multiply('x', 'x')), {'x': 2})
        7  # because 3+2^2 = 7

    Input:
        expr: An expression. The expression can be the following
            (1) any real number
            (2) 'x'
            (3) operation(expr, expr), where operation can be either add or multiply
        value_map: A dictionary specifying the values of the variables.

    Output:
        This returns expr evaluated at the specific value of x, which should be  a
            real number.
    """

    try:
        return expr.evaluate(value_map)
    except AttributeError:
        if expr in value_map:
            return value_map[expr]
        else:
            return expr


def differentiate(expr, value_map):
    """Compute derivative of expr with respect to x for a specific value of x.

    Examples:
        >>> differentiate(multiply(3, 'x'), {'x': 0})
            3  # because d(3x)/d(x) = 3
        >>> differentiate(add(3, multiply('x', 'x')), {'x': 2})
            4  # because d(3+x^2)/d(x) = 2x -> 2*2 = 4

    Input:
        expr: An expression. The expression can be the following
            (1) any real number
            (2) 'x'
            (3) operation(expr, expr), where operation can be either add or multiply
        value_map: A dictionary specifying the values of the variables.

    Output:
        This returns the derivative of expr with respect to x evaluated at the
            Specific value of x, which should be a real number.
    """
    try:
        return expr.diff(value_map)
    except AttributeError:
        if expr == 'x':
            return 1
        else:
            return 0


class add:
    def __init__(self, expr1, expr2):
        self.in1 = expr1
        self.in2 = expr2
        self.local_grad1 = 1
        self.local_grad2 = 1

    def evaluate(self, val_map):
        try:
            self.in1 = self.in1.evaluate(val_map)
        except AttributeError:
            if self.in1 in val_map:
                self.in1 = val_map[self.in1]
        try:
            self.in2 = self.in2.evaluate(val_map)
        except AttributeError:
            if self.in2 in val_map:
                self.in2 = val_map[self.in2]
        return self.in1 + self.in2

    def diff(self, val_map):
        # diff of x + y = dx + dy
        try:
            self.local_grad1 = self.in1.diff(val_map)
        except AttributeError:
            if self.in1 == 'x':
                self.local_grad1 = 1
            else:
                self.local_grad1 = 0
        try:
            self.local_grad2 = self.in2.diff(val_map)
        except AttributeError:
            if self.in2 == 'x':
                self.local_grad2 = 1
            else:
                self.local_grad2 = 0
        return self.local_grad1 + self.local_grad2


class multiply:
    def __init__(self, expr1, expr2):
        self.in1 = expr1
        self.in2 = expr2
        self.local_grad1 = self.in2
        self.local_grad2 = self.in1

    def evaluate(self, val_map):
        try:
            self.in1 = self.in1.evaluate(val_map)
        except AttributeError:
            if self.in1 in val_map:
                self.in1 = val_map[self.in1]
        try:
            self.in2 = self.in2.evaluate(val_map)
        except AttributeError:
            if self.in2 in val_map:
                self.in2 = val_map[self.in2]
        return self.in1 * self.in2

    def diff(self, val_map):
        # derivative of x*y = x*dy + dx*y
        try:
            self.local_grad1 = self.in1.diff(val_map)
            self.in1 = self.in1.evaluate(val_map)
        except AttributeError:
            if self.in1 == 'x':
                self.local_grad1 = 1
                self.in1 = val_map[self.in1]
            else:
                self.local_grad1 = 0
                if self.in1 in val_map:
                    self.in1 = val_map[self.in1]
        try:
            self.local_grad2 = self.in2.diff(val_map)
            self.in2 = self.in2.evaluate(val_map)
        except AttributeError:
            if self.in2 == 'x':
                self.local_grad2 = 1
                self.in2 = val_map[self.in2]
            else:
                self.local_grad2 = 0
                if self.in2 in val_map:
                    self.in2 = val_map[self.in2]
        return self.local_grad1*self.in2 + self.local_grad2*self.in1


print(evaluate(add(3, 'x'), {'x': 2, 'y': 5}))
print(differentiate(add(3, 'x'), {'x': 2, 'y': 5}))
print(differentiate(add(3, 3), {'x': 2, 'y': 5}))
print(differentiate(add('y', 'x'), {'x': 2, 'y': 5}))
print(differentiate(add('x', 'x'), {'x': 2, 'y': 5}))
print(differentiate(multiply(3, 3), {'x': 2, 'y': 5}))
print(differentiate(multiply(3, 'x'), {'x': 2, 'y': 5}))
print(differentiate(multiply('x', 'y'), {'x': 2, 'y': 5}))
print(differentiate(multiply('x', 'x'), {'x': 2, 'y': 5}))
print(differentiate(multiply('x', 'x'), {'x': 3, 'y': 5}))
print(differentiate(add(3, multiply('x', 'x')), {'x': 2, 'y': 5}))
print(differentiate(add(3, add(multiply(2, 'x'), multiply('x', multiply('x', 'x')))), {'x': 2, 'y': 5}))
