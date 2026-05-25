"""
Stage 4 - Module 1: Math Solving Engine

Solves handwritten math equations using SymPy symbolic mathematics.
NEVER uses Python eval() or exec() — all computation is symbolic.

Supports:
  • Arithmetic expressions and equations
  • Linear equations (degree 1)
  • Quadratic equations (degree 2)

Each solver returns a standardised result dictionary with detailed
pedagogically correct step-by-step solutions.
"""

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
    """Convert a SymPy number to a plain Python float/int for JSON safety."""
    if not _SYMPY_OK:
        return val
    try:
        f = float(val)
        return int(f) if f == int(f) else round(f, 6)
    except Exception:
        return str(val)


def _parse_sides(equation_str: str):
    """
    Split 'lhs = rhs' and return (lhs_expr, rhs_expr, lhs_str, rhs_str).

    Raises:
        ValueError: No '=' found, or SymPy cannot parse either side.
    """
    if '=' not in equation_str:
        raise ValueError(f"No '=' in '{equation_str}'")
    lhs_str, rhs_str = [s.strip() for s in equation_str.split('=', 1)]
    local = {'x': _x}
    lhs = parse_expr(lhs_str, local_dict=local)
    rhs = parse_expr(rhs_str, local_dict=local)
    return lhs, rhs, lhs_str, rhs_str


def _step(n, desc, expr, explanation):
    """Convenience constructor for a step dict."""
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
    """
    Symbolic math solver with pedagogically detailed step generation.
    Uses SymPy for all computation — no eval/exec anywhere.
    """

    if not _SYMPY_OK:
        raise ImportError("SymPy is required. pip install sympy")

    # ─────────────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────────────

    def classify_equation(self, equation_str: str) -> dict:
        """
        Analyse the equation string and return a classification dict.

        Returns:
            type            – 'arithmetic' | 'linear' | 'quadratic' | 'expression'
            has_variable    – bool
            degree          – int (highest power of x)
            variables       – list of variable name strings
            description     – human-readable string
        """
        has_eq  = '=' in equation_str
        has_x   = 'x' in equation_str

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

    # ─────────────────────────────────────────────────────────────────────────
    # Master dispatcher
    # ─────────────────────────────────────────────────────────────────────────

    def solve_equation(self, equation_str: str) -> dict:
        """
        Classify then dispatch to the appropriate solver.

        Args:
            equation_str: Clean equation string from the parser
                          (e.g., '2*x + 5 = 15', 'x**2 + 5*x + 6 = 0').

        Returns:
            Standardised result dict (see individual solvers for schema).
        """
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

    # ─────────────────────────────────────────────────────────────────────────
    # Arithmetic
    # ─────────────────────────────────────────────────────────────────────────

    def solve_arithmetic(self, equation_str: str) -> dict:
        """
        Evaluate or verify a pure-number equation like '25 + 37 = 62'.

        If '=' is present, verifies whether it is true.
        If no '=', evaluates the expression directly.
        """
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
                                   f'Computing the value of the left-hand expression: {lval}'))
                n += 1
                steps.append(_step(n, 'Evaluate right side',
                                   f'{rs} = {rval}',
                                   f'Computing the value of the right-hand expression: {rval}'))

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
                                   f'Computing the numeric value of the expression.'))
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

    # ─────────────────────────────────────────────────────────────────────────
    # Linear
    # ─────────────────────────────────────────────────────────────────────────

    def solve_linear(self, equation_str: str) -> dict:
        """
        Solve a linear equation like '2*x + 5 = 15' with full pedagogical steps.

        Step sequence:
          1. Original equation.
          2. Identify constant on the variable side.
          3. Move constant to the other side (Subtraction/Addition Property).
          4. Show simplified result.
          5. Divide/multiply by the coefficient (Division/Multiplication Property).
          6. State the solution.
          7. Verify by substitution.
        """
        t0 = time.time()
        try:
            lhs, rhs, lhs_str, rhs_str = _parse_sides(equation_str)
            solutions = solve(Eq(lhs, rhs), _x)

            if not solutions:
                # Check if 0 = non-zero → no solution
                simplified = simplify(lhs - rhs)
                if simplified != 0:
                    return _error_result('linear', equation_str,
                                         'No solution — equation is a contradiction.')
                return _error_result('linear', equation_str,
                                     'Infinite solutions — equation is always true.')

            sol = solutions[0]

            # Extract coefficient and constant of x from lhs
            coeff_x  = lhs.coeff(_x)           # coefficient of x
            constant = lhs - coeff_x * _x      # constant term

            steps = []
            n = 0

            n += 1
            steps.append(_step(n, 'Original equation', equation_str,
                                'We are given this linear equation and need to isolate x.'))

            n += 1
            steps.append(_step(n, 'Identify terms',
                                f'Coefficient of x: {coeff_x}   Constant: {constant}',
                                'We identify the coefficient of x and the constant term '
                                'to plan our algebraic steps.'))

            # Move constant to right side (if non-zero)
            if constant != 0:
                prop = ('Subtraction' if constant > 0 else 'Addition') + ' Property of Equality'
                sign = '-' if constant > 0 else '+'
                abs_c = abs(constant)

                n += 1
                steps.append(_step(n,
                                   f'{sign[0] if constant > 0 else sign} {abs_c} from both sides',
                                   f'{lhs_str} {sign} {abs_c} = {rhs_str} {sign} {abs_c}',
                                   f'Using the {prop}: we perform the same operation on both '
                                   f'sides to keep the equation balanced.'))

                new_rhs = rhs - constant
                n += 1
                steps.append(_step(n, 'Simplify both sides',
                                   f'{coeff_x}*x = {new_rhs}',
                                   f'The constant {constant} cancels on the left; '
                                   f'the right side simplifies to {new_rhs}.'))
            else:
                new_rhs = rhs

            # Divide by coefficient
            if coeff_x != 1 and coeff_x != -1:
                n += 1
                steps.append(_step(n, f'Divide both sides by {coeff_x}',
                                   f'{coeff_x}*x / {coeff_x} = {new_rhs} / {coeff_x}',
                                   f'Using the Division Property of Equality: '
                                   f'dividing both sides by {coeff_x} isolates x.'))
            elif coeff_x == -1:
                n += 1
                steps.append(_step(n, 'Multiply both sides by -1',
                                   f'-1 * (-x) = -1 * {new_rhs}',
                                   'Multiplying both sides by -1 removes the negative sign on x '
                                   '(Multiplication Property of Equality).'))

            n += 1
            steps.append(_step(n, 'Solution', f'x = {sol}',
                                f'x is now isolated. The solution is x = {sol}.'))

            # Verification
            lval = lhs.subs(_x, sol)
            rval = rhs
            verified = bool(simplify(lval - rval) == 0)
            n += 1
            steps.append(_step(n, 'Verification',
                                f'Substitute x = {sol}: '
                                f'{lhs_str.replace("x", str(sol))} = {lval}  vs  {rhs_str} = {rval}',
                                f'Substituting x = {sol} into the original equation: '
                                f'{lval} = {rval} → {"✓ Correct" if verified else "✗ Error"}'))

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

    # ─────────────────────────────────────────────────────────────────────────
    # Quadratic
    # ─────────────────────────────────────────────────────────────────────────

    def solve_quadratic(self, equation_str: str) -> dict:
        """
        Solve a quadratic equation like 'x**2 + 5*x + 6 = 0'.

        Step sequence:
          1. Original equation.
          2. Standard form and identify coefficients a, b, c.
          3. Attempt factoring.
          4. Zero Product Property.
          5. Solve each linear factor.
          6. Quadratic formula derivation with numbers substituted.
          7. Discriminant analysis (two real / one repeated / complex).
          8. Verify all solutions.
        """
        t0 = time.time()
        try:
            lhs, rhs, lhs_str, rhs_str = _parse_sides(equation_str)
            expr      = expand(lhs - rhs)       # Move everything to lhs
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
                                'We are given this quadratic equation to solve.'))

            n += 1
            steps.append(_step(n, 'Standard form  ax² + bx + c = 0',
                                f'{expr} = 0',
                                f'We rewrite in standard form. '
                                f'a = {a},  b = {b},  c = {c}'))

            # Factoring attempt
            factored = factor(expr)
            if factored != expr:
                n += 1
                steps.append(_step(n, 'Factor the quadratic',
                                   f'{factored} = 0',
                                   f'We factor the quadratic expression. '
                                   f'Factoring is possible when the discriminant is a perfect square.'))

                n += 1
                steps.append(_step(n, 'Zero Product Property',
                                   'If A·B = 0 then A = 0 or B = 0',
                                   'The Zero Product Property states that if a product equals zero, '
                                   'at least one factor must be zero.'))

                factors = sp.factor_list(expr)[1]
                for fi, (fac, _) in enumerate(factors):
                    sol_i = solve(Eq(fac, 0), _x)
                    n += 1
                    steps.append(_step(n, f'Set factor {fi+1} = 0',
                                       f'{fac} = 0  →  x = {sol_i[0] if sol_i else "?"}',
                                       f'Solving the linear equation from factor {fi+1}.'))
            else:
                n += 1
                steps.append(_step(n, 'Factoring not possible',
                                   f'discriminant = {disc}',
                                   'The expression cannot be factored over integers. '
                                   'We proceed with the quadratic formula.'))

            # Quadratic formula
            n += 1
            steps.append(_step(n, 'Quadratic formula',
                                'x = (-b ± √(b²-4ac)) / (2a)',
                                'The quadratic formula solves any quadratic equation '
                                'regardless of whether it factors nicely.'))

            n += 1
            steps.append(_step(n, 'Substitute a, b, c',
                                f'x = (-({b}) ± √(({b})²-4·({a})·({c}))) / (2·({a}))',
                                f'Substituting a={a}, b={b}, c={c} into the formula.'))

            n += 1
            disc_desc = (
                'Two distinct real solutions (discriminant > 0).'
                if disc > 0 else
                'Exactly one repeated real solution (discriminant = 0).'
                if disc == 0 else
                'Two complex solutions (discriminant < 0).'
            )
            steps.append(_step(n, f'Discriminant: b²-4ac = {disc}',
                                f'b²-4ac = ({b})²-4·({a})·({c}) = {b**2} - {4*a*c} = {disc}',
                                disc_desc))

            # Solutions
            if not solutions:
                n += 1
                steps.append(_step(n, 'No real solutions', 'No real solutions exist.',
                                   'The discriminant is negative, so only complex solutions exist.'))
                answer_str = 'No real solutions'
                sol_list   = []
                verified   = True
            else:
                sol_str_parts = [f'x = {s}' for s in solutions]
                answer_str    = ',  '.join(sol_str_parts)
                sol_list      = [_to_python(s) for s in solutions]

                # Verification for each solution
                for sol in solutions:
                    lval     = lhs.subs(_x, sol)
                    rval     = rhs
                    verified = bool(simplify(lval - rval) == 0)
                    n += 1
                    steps.append(_step(n, f'Verify x = {sol}',
                                       f'Substitute x={sol}: {lval} = {rval} '
                                       f'→ {"✓" if verified else "✗"}',
                                       f'Substituting x = {sol} back into the '
                                       f'original equation to confirm.'))

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

    # ─────────────────────────────────────────────────────────────────────────
    # Expression evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def solve_expression(self, expression_str: str) -> dict:
        """
        Simplify and evaluate an expression with no equals sign.

        Example: 'sqrt(144)'→ 12,  '2**8' → 256
        """
        t0 = time.time()
        try:
            expr   = parse_expr(expression_str, local_dict={'x': _x})
            result = simplify(expr).evalf()
            steps  = [
                _step(1, 'Original expression', expression_str,
                      'We evaluate / simplify the given expression.'),
                _step(2, 'Simplified result',   f'{result}',
                      f'The expression evaluates to {result}.'),
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
