

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
    # TODO


def add(expr1, expr2):
    pass


def multiply(expr1, expr2):
    pass


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
    # TODO
