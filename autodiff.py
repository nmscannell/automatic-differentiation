"""
Auto-diff for Anyscale

Natasha Scannell
"""


def evaluate(expr, value_map):
    """Evaluate expr for a specific value of x.

    Examples:
    #    >>> evaluate(multiply(3, 'x'), {'x':2})
            0  # because 3*0 = 0
    #    >>> evaluate(add(3, multiply('x', 'x')), {'x': 2})
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
    #    >>> differentiate(multiply(3, 'x'), {'x': 0})
            3  # because d(3x)/d(x) = 3
    #    >>> differentiate(add(3, multiply('x', 'x')), {'x': 2})
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
    """ The add operation can take two inputs and calculate
    the value of the add function evaluated on the two input.
    It can also calculate the derivative of an add unit
    with respect to 'x'.

    Input:
        expr1 and expr2: Expressions. These expressions can be the following:
            (1) any real number
            (2) 'x'
            (3) operation(expr, expr), where operation can be either add or multiply
    """
    def __init__(self, expr1, expr2):
        self.in1 = expr1
        self.in2 = expr2
        self.local_grad1 = 1
        self.local_grad2 = 1

    def evaluate(self, val_map):
        """ Computes the add function on the two inputs, self.in1 and self.in2.

        Example:
        #    >>> self.in1 = 3, self.in2 = x, self.evaluate({'x': 4})
                7  # because 3 + 4 = 7

        Input:
            value_map: A dictionary specifying the values of variables.

        Output:
            The result of self.in1 + self.in2
        """
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
        """ Computes the derivative of addition of two inputs, self.in1 and self.in2
        with respect to 'x'. d/dx[self.in1 + self.in2] = d(self.in1)/dx + d(self.in2)/dx
        --The derivative of a constant or variable that is not 'x' with respect to 'x' is 0.
        --The derivative of 'x' with respect to 'x' is 1.
        --If one of the inputs is an operation, its derivative must be found.

        Examples:
        #    >>> self.in1 = 3, self.in2 = x, self.diff({'x': 4})
                1  # because d/dx[3 + x] = 0 + 1 = 1

        #    >>> self.in1 = x, self.in2 = x, self.diff({'x': 4})
                2  # because d/dx[x + x] = 1 + 1 = 2

        Input:
            value_map: A dictionary specifying the values of variables.

        Output:
            The result of d/dx[self.in1 + self.in2]
        """
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
                print('input is x')
                self.local_grad2 = 1
            else:
                self.local_grad2 = 0
        return self.local_grad1 + self.local_grad2


class multiply:
    """ The multiply operation can take two inputs and calculate
    the value of the multiply function evaluated on the two input.
    It can also calculate the derivative of a multiply operation
    with respect to 'x'.

    Input:
        expr1 and expr2: Expressions. These expressions can be the following:
            (1) any real number
            (2) 'x'
            (3) operation(expr, expr), where operation can be either add or multiply
    """
    def __init__(self, expr1, expr2):
        self.in1 = expr1
        self.in2 = expr2
        self.local_grad1 = self.in2
        self.local_grad2 = self.in1

    def evaluate(self, val_map):
        """ Computes the multiply function on the two inputs, self.in1 and self.in2.

        Example:
        #    >>> self.in1 = 3, self.in2 = x, self.evaluate({'x': 4})
                12  # because 3 * 4 = 12

        Input:
            value_map: A dictionary specifying the values of variables.

        Output:
            The result of self.in1 * self.in2
        """
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
        """ Computes the derivative of multiplication of two inputs, self.in1 and self.in2
        with respect to 'x'. By the product rule,
        d/dx[self.in1 * self.in2] = (d(self.in1)/dx) * self.in2 + (d(self.in2)/dx) * self.in1
        --The derivative of a constant or variable that is not 'x' with respect to 'x' is 0.
        --The derivative of 'x' with respect to 'x' is 1.
        --If one of the inputs is an operation, its derivative must be found.
        --If one of the inputs is an operation or in the value map, it must be evaluated and updated.

        Examples:
        #    >>> self.in1 = 3, self.in2 = x, self.diff({'x': 4})
                3  # because d/dx[3 * x] = d/dx[3] * x + d/dx[x] * 3 = 0 + 3 = 3

        #    >>> self.in1 = x, self.in2 = x, self.diff({'x': 4})
                8  # because d/dx[x * x] = d/dx[x] * x + d/dx[x] * x = x + x = 8

        #    >>> self.in1 = x, self.in2 = x * x, self.diff({'x': 4})
                48  # because x * x^2 = x^3. d/dx[x^3] = 3x^2. evaluated at 'x' = 4: 3*(4)^2 = 48
                d/dx[x * x^2] = d/dx[x] * x^2 + d/dx[x^2] * x = 1 * x^2 + [2x] * x = x^2 + 2x^2 = 3x^2

        Input:
            value_map: A dictionary specifying the values of variables.

        Output:
            The result of d/dx[self.in1 * self.in2]
        """
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
