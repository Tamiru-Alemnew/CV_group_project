import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sympy import (
        symbols, parse_expr, Eq, solve, factor, expand, simplify,
        Symbol, Rational, sqrt, latex as sympy_latex,
        discriminant as sympy_discriminant, Poly, SympifyError,
        oo, zoo, nan, im
    )
    import sympy as sp
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

_x = symbols('x') if _SYMPY_OK else None


def _to_python(val):
    """Convert a SymPy number to a plain Python float/int."""
    if not _SYMPY_OK:
        return val
    try:
        f = float(val)
        return int(f) if f == int(f) else round(f, 6)
    except Exception:
        return str(val)


def _parse_sides(equation_str: str):
    if '=' not in equation_str:
        raise ValueError(f"No '=' in '{equation_str}'")
    lhs_str, rhs_str = [s.strip() for s in equation_str.split('=', 1)]
    local = {'x': _x}
    lhs = parse_expr(lhs_str, local_dict=local)
    rhs = parse_expr(rhs_str, local_dict=local)
    return lhs, rhs, lhs_str, rhs_str


def _step(n, desc, expr, explanation):
    return {
        'step_number': n,
        'description': desc,
        'expression':  str(expr),
        'explanation': explanation,
    }


def _error_result(eq_type, equation_str, msg):
    return {
        'success':            False,
        'equation_type':      eq_type,
        'original_equation':  equation_str,
        'formatted_equation': equation_str,
        'solution':           [],
        'answer_str':         '',
        'steps':              [],
        'verification':       {'verified': False},
        'error':              msg,
        'metadata':           {'num_steps': 0, 'solve_time_ms': 0, 'equation_degree': 0},
    }


class MathSolver:

    if not _SYMPY_OK:
        raise ImportError("SymPy is required. pip install sympy")

    def classify_equation(self, equation_str: str) -> dict:
        has_eq = '=' in equation_str
        has_x  = 'x' in equation_str

        if not has_eq and not has_x:
            return {'type': 'expression',  'has_variable': False, 'degree': 0,
                    'variables': [], 'description': 'Arithmetic expression to evaluate'}
        if not has_x:
            return {'type': 'arithmetic', 'has_variable': False, 'degree': 0,
                    'variables': [], 'description': 'Arithmetic equation to verify'}

        try:
            lhs, rhs, _, _ = _parse_sides(equation_str) if has_eq else (
                parse_expr(equation_str, local_dict={'x': _x}), sp.Integer(0), None, None)
            expr = lhs - rhs
            poly = Poly(expr, _x)
            deg  = poly.degree()
        except Exception:
            deg = 1 if 'x**2' not in equation_str else 2

        if deg == 1:
            return {'type': 'linear',    'has_variable': True, 'degree': 1,
                    'variables': ['x'], 'description': 'Linear equation (degree 1)'}
        if deg == 2:
            return {'type': 'quadratic', 'has_variable': True, 'degree': 2,
                    'variables': ['x'], 'description': 'Quadratic equation (degree 2)'}

        return {'type': 'linear', 'has_variable': True, 'degree': deg,
                'variables': ['x'], 'description': f'Polynomial degree {deg}'}

    def solve_equation(self, equation_str: str) -> dict:
        eq_str = equation_str.strip()
        info   = self.classify_equation(eq_str)

        if info['type'] == 'expression':
            return self.solve_expression(eq_str)
        if info['type'] == 'arithmetic':
            return self.solve_arithmetic(eq_str)
        if info['type'] == 'linear':
            return self.solve_linear(eq_str)
        if info['type'] == 'quadratic':
            return self.solve_quadratic(eq_str)
        return _error_result('unknown', eq_str, f"Unsupported type: {info['type']}")

    def solve_arithmetic(self, equation_str: str) -> dict:
        t0 = time.time()
        try:
            steps = []
            n = 0

            n += 1
            steps.append(_step(n, 'Original equation', equation_str,
                                'Start with the given arithmetic statement.'))

            if '=' in equation_str:
                lhs, rhs, ls, rs = _parse_sides(equation_str)
                lval = lhs.evalf()
                rval = rhs.evalf()

                n += 1
                steps.append(_step(n, 'Evaluate left side',
                                   f'{ls} = {lval}',
                                   f'Computing the left-hand expression: {lval}'))
                n += 1
                steps.append(_step(n, 'Evaluate right side',
                                   f'{rs} = {rval}',
                                   f'Computing the right-hand expression: {rval}'))

                verified = bool(simplify(lhs - rhs) == 0)
                n += 1
                steps.append(_step(n, 'Check equality',
                                   f'{lval} = {rval}  →  {"TRUE ✓" if verified else "FALSE ✗"}',
                                   f'The equation is {"correct" if verified else "incorrect"}.'))

                answer_str = f'{lval} = {rval} is {"TRUE" if verified else "FALSE"}'
                sol_val    = _to_python(lval)
            else:
                expr = parse_expr(equation_str, local_dict={'x': _x})
                result = expr.evalf()
                n += 1
                steps.append(_step(n, 'Evaluate expression',
                                   f'{equation_str} = {result}',
                                   f'Computing the numeric value.'))
                answer_str = f'{equation_str} = {result}'
                sol_val    = _to_python(result)
                verified   = True

            elapsed = (time.time() - t0) * 1000
            return {
                'success':            True,
                'equation_type':      'arithmetic',
                'original_equation':  equation_str,
                'formatted_equation': equation_str,
                'solution':           [sol_val],
                'answer_str':         answer_str,
                'steps':              steps,
                'verification':       {'verified': verified},
                'error':              None,
                'metadata':           {'num_steps': len(steps),
                                       'solve_time_ms': round(elapsed, 2),
                                       'equation_degree': 0},
            }
        except Exception as e:
            return _error_result('arithmetic', equation_str, str(e))

    def solve_linear(self, equation_str: str) -> dict:
        t0 = time.time()
        try:
            lhs, rhs, lhs_str, rhs_str = _parse_sides(equation_str)
            solutions = solve(Eq(lhs, rhs), _x)

            if not solutions:
                simplified = simplify(lhs - rhs)
                if simplified != 0:
                    return _error_result('linear', equation_str,
                                         'No solution — equation is a contradiction.')
                return _error_result('linear', equation_str,
                                     'Infinite solutions — equation is always true.')

            sol      = solutions[0]
            coeff_x  = lhs.coeff(_x)
            constant = lhs - coeff_x * _x

            steps = []
            n = 0

            n += 1
            steps.append(_step(n, 'Original equation', equation_str,
                                'Isolate x.'))

            n += 1
            steps.append(_step(n, 'Identify terms',
                                f'Coefficient of x: {coeff_x}   Constant: {constant}',
                                'Identify coefficient and constant term.'))

            if constant != 0:
                prop = ('Subtraction' if constant > 0 else 'Addition') + ' Property of Equality'
                sign = '-' if constant > 0 else '+'
                abs_c = abs(constant)

                n += 1
                steps.append(_step(n,
                                   f'{sign} {abs_c} from both sides',
                                   f'{lhs_str} {sign} {abs_c} = {rhs_str} {sign} {abs_c}',
                                   f'Using the {prop}.'))

                new_rhs = rhs - constant
                n += 1
                steps.append(_step(n, 'Simplify',
                                   f'{coeff_x}*x = {new_rhs}',
                                   f'Constant cancels on the left; right simplifies to {new_rhs}.'))
            else:
                new_rhs = rhs

            if coeff_x != 1 and coeff_x != -1:
                n += 1
                steps.append(_step(n, f'Divide both sides by {coeff_x}',
                                   f'{coeff_x}*x / {coeff_x} = {new_rhs} / {coeff_x}',
                                   f'Division Property of Equality.'))
            elif coeff_x == -1:
                n += 1
                steps.append(_step(n, 'Multiply both sides by -1',
                                   f'-1 * (-x) = -1 * {new_rhs}',
                                   'Remove the negative sign on x.'))

            n += 1
            steps.append(_step(n, 'Solution', f'x = {sol}', f'x = {sol}'))

            lval     = lhs.subs(_x, sol)
            rval     = rhs
            verified = bool(simplify(lval - rval) == 0)
            n += 1
            steps.append(_step(n, 'Verification',
                                f'Substitute x = {sol}: {lval} = {rval} '
                                f'→ {"✓" if verified else "✗"}',
                                f'Substituting x = {sol} into the original equation.'))

            elapsed = (time.time() - t0) * 1000
            return {
                'success':            True,
                'equation_type':      'linear',
                'original_equation':  equation_str,
                'formatted_equation': f'{lhs_str} = {rhs_str}',
                'solution':           [_to_python(sol)],
                'answer_str':         f'x = {sol}',
                'steps':              steps,
                'verification':       {
                    'verified':    verified,
                    'left_value':  _to_python(lval),
                    'right_value': _to_python(rval),
                },
                'error':   None,
                'metadata': {'num_steps': len(steps),
                             'solve_time_ms': round(elapsed, 2),
                             'equation_degree': 1},
            }

        except Exception as e:
            return _error_result('linear', equation_str, str(e))

    def solve_quadratic(self, equation_str: str) -> dict:
        t0 = time.time()
        try:
            lhs, rhs, lhs_str, rhs_str = _parse_sides(equation_str)
            expr      = expand(lhs - rhs)
            solutions = solve(Eq(lhs, rhs), _x)

            poly = Poly(expr, _x)
            a    = poly.nth(2)
            b    = poly.nth(1)
            c    = poly.nth(0)
            disc = b**2 - 4*a*c

            steps = []
            n = 0

            n += 1
            steps.append(_step(n, 'Original equation', equation_str,
                                'Solve this quadratic equation.'))

            n += 1
            steps.append(_step(n, 'Standard form  ax² + bx + c = 0',
                                f'{expr} = 0',
                                f'a = {a},  b = {b},  c = {c}'))

            factored = factor(expr)
            if factored != expr:
                n += 1
                steps.append(_step(n, 'Factor the quadratic',
                                   f'{factored} = 0',
                                   'Expression factors over integers.'))

                n += 1
                steps.append(_step(n, 'Zero Product Property',
                                   'If A·B = 0 then A = 0 or B = 0',
                                   'At least one factor must be zero.'))

                factors = sp.factor_list(expr)[1]
                for fi, (fac, _) in enumerate(factors):
                    sol_i = solve(Eq(fac, 0), _x)
                    n += 1
                    steps.append(_step(n, f'Set factor {fi+1} = 0',
                                       f'{fac} = 0  →  x = {sol_i[0] if sol_i else "?"}',
                                       f'Solving factor {fi+1}.'))
            else:
                n += 1
                steps.append(_step(n, 'Factoring not possible',
                                   f'discriminant = {disc}',
                                   'Proceed with the quadratic formula.'))

            n += 1
            steps.append(_step(n, 'Quadratic formula',
                                'x = (-b ± √(b²-4ac)) / (2a)',
                                'Solves any quadratic equation.'))

            n += 1
            steps.append(_step(n, 'Substitute a, b, c',
                                f'x = (-({b}) ± √(({b})²-4·({a})·({c}))) / (2·({a}))',
                                f'a={a}, b={b}, c={c}'))

            n += 1
            disc_desc = (
                'Two distinct real solutions.' if disc > 0 else
                'One repeated real solution.'  if disc == 0 else
                'Two complex solutions.'
            )
            steps.append(_step(n, f'Discriminant: {disc}',
                                f'b²-4ac = {b**2} - {4*a*c} = {disc}',
                                disc_desc))

            if not solutions:
                n += 1
                steps.append(_step(n, 'No real solutions', 'No real solutions exist.',
                                   'Discriminant is negative.'))
                answer_str = 'No real solutions'
                sol_list   = []
            else:
                answer_str = ',  '.join(f'x = {s}' for s in solutions)
                sol_list   = [_to_python(s) for s in solutions]

                for sol in solutions:
                    lval     = lhs.subs(_x, sol)
                    rval     = rhs
                    verified = bool(simplify(lval - rval) == 0)
                    n += 1
                    steps.append(_step(n, f'Verify x = {sol}',
                                       f'{lval} = {rval} → {"✓" if verified else "✗"}',
                                       f'Substituting x = {sol} back into the original equation.'))

            elapsed = (time.time() - t0) * 1000
            return {
                'success':            True,
                'equation_type':      'quadratic',
                'original_equation':  equation_str,
                'formatted_equation': f'{expr} = 0',
                'solution':           sol_list,
                'answer_str':         answer_str,
                'steps':              steps,
                'verification':       {'verified': True},
                'error':              None,
                'metadata':           {'num_steps': len(steps),
                                       'solve_time_ms': round(elapsed, 2),
                                       'equation_degree': 2},
            }

        except Exception as e:
            return _error_result('quadratic', equation_str, str(e))

    def solve_expression(self, expression_str: str) -> dict:
        t0 = time.time()
        try:
            expr   = parse_expr(expression_str, local_dict={'x': _x})
            result = simplify(expr).evalf()
            steps  = [
                _step(1, 'Original expression', expression_str, 'Evaluate the expression.'),
                _step(2, 'Result', f'{result}', f'= {result}'),
            ]
            elapsed = (time.time() - t0) * 1000
            return {
                'success':            True,
                'equation_type':      'expression',
                'original_equation':  expression_str,
                'formatted_equation': expression_str,
                'solution':           [_to_python(result)],
                'answer_str':         f'{expression_str} = {result}',
                'steps':              steps,
                'verification':       {'verified': True},
                'error':              None,
                'metadata':           {'num_steps': 2,
                                       'solve_time_ms': round(elapsed, 2),
                                       'equation_degree': 0},
            }
        except Exception as e:
            return _error_result('expression', expression_str, str(e))
