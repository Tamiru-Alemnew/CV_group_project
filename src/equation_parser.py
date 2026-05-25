"""
Stage 3 - Module 2: Equation Parser

Converts the ordered list of recognised symbol dicts into a clean
mathematical equation string that SymPy can parse and solve.

Transformations applied in order:
  1. Exponent handling  (superscript position → ** notation)
  2. Digit merging      (['1','5'] → '15')
  3. Implicit multiply  (['2','x'] → ['2','*','x'])
  4. Negative handling  (leading minus, minus-after-operator)
  5. String assembly    (spacing around +, -, =)
"""

import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    from sympy import parse_expr, symbols, sympify, SympifyError
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False


class EquationParser:
    """
    Transforms a sequence of recognised symbol dicts into a SymPy-parseable
    equation string, logging every intermediate transformation step.
    """

    OPERATOR_SYMBOLS = {'+', '-', '*', '/', '='}
    DIGIT_SYMBOLS    = set('0123456789')

    def __init__(self):
        self._steps: dict = {}   # Records intermediate token lists

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def parse(self, recognized_symbols) -> tuple[str, dict]:
        """
        Main parse entry-point.

        Args:
            recognized_symbols: Either:
                • List of recognition result dicts (full pipeline use), each
                  containing at least 'symbol' and optionally 'position_type'.
                • List of plain strings (direct testing shortcut).

        Returns:
            (equation_str, transformation_steps)
            where transformation_steps is a dict of token lists at each stage.
        """
        self._steps = {}

        # Normalise input to list of (symbol, position_type) tuples
        if recognized_symbols and isinstance(recognized_symbols[0], dict):
            symbols_raw = [r['symbol']                          for r in recognized_symbols]
            types_raw   = [r.get('position_type', 'normal')    for r in recognized_symbols]
        else:
            symbols_raw = list(recognized_symbols)
            types_raw   = ['normal'] * len(symbols_raw)

        self._steps['raw'] = symbols_raw.copy()

        # Step 1: Exponents
        tokens, types = self.handle_exponents(symbols_raw, types_raw)
        self._steps['after_exponents'] = tokens.copy()

        # Step 2: Merge consecutive digits
        tokens = self.merge_digits(tokens)
        self._steps['after_digit_merge'] = tokens.copy()

        # Step 3: Implicit multiplication
        tokens = self.insert_implicit_multiply(tokens)
        self._steps['after_implicit_mult'] = tokens.copy()

        # Step 4: Negative numbers
        tokens = self.handle_negatives(tokens)
        self._steps['after_negatives'] = tokens.copy()

        # Step 5: Build string
        equation_str = self._build_string(tokens)

        return equation_str, self._steps

    # ─────────────────────────────────────────────────────────────────────────
    # Transformation steps
    # ─────────────────────────────────────────────────────────────────────────

    def handle_exponents(self, symbols: list[str],
                          types: list[str]) -> tuple[list[str], list[str]]:
        """
        Merge [base, superscript_digit] pairs into 'base**digit' tokens.

        Why: A superscript character (position_type='superscript') from the
        segmenter represents an exponent. Python and SymPy use ** for
        exponentiation so we emit that directly.

        Example:
            ['x', '2'(superscript)] → ['x**2']
        """
        result_syms  = []
        result_types = []
        i = 0
        while i < len(symbols):
            if (i + 1 < len(symbols)
                    and types[i + 1] == 'superscript'
                    and symbols[i + 1] in self.DIGIT_SYMBOLS
                    and symbols[i] not in self.OPERATOR_SYMBOLS):
                # Collect all consecutive superscript digits for multi-digit exponents
                exp_digits = []
                j = i + 1
                while j < len(symbols) and types[j] == 'superscript':
                    exp_digits.append(symbols[j])
                    j += 1
                result_syms.append(f"{symbols[i]}**{''.join(exp_digits)}")
                result_types.append(types[i])
                i = j
            else:
                result_syms.append(symbols[i])
                result_types.append(types[i])
                i += 1
        return result_syms, result_types

    def merge_digits(self, tokens: list[str]) -> list[str]:
        """
        Merge consecutive single-digit tokens into multi-digit numbers.

        Why: Individual digits like ['1', '5'] represent the number 15 when
        they appear side-by-side. Without merging, '15' would be parsed as
        two separate operands, breaking arithmetic.

        Example:
            ['1', '5', '+', '3'] → ['15', '+', '3']
        """
        result = []
        i = 0
        while i < len(tokens):
            # Start a digit run
            if (len(tokens[i]) == 1 and tokens[i] in self.DIGIT_SYMBOLS):
                run = tokens[i]
                j   = i + 1
                while j < len(tokens) and len(tokens[j]) == 1 and tokens[j] in self.DIGIT_SYMBOLS:
                    run += tokens[j]
                    j   += 1
                result.append(run)
                i = j
            else:
                result.append(tokens[i])
                i += 1
        return result

    def insert_implicit_multiply(self, tokens: list[str]) -> list[str]:
        """
        Insert '*' between tokens where handwritten convention implies multiplication.

        Cases handled:
          • number  before variable  :  2x   → 2*x
          • number  before open-paren:  3(   → 3*(
          • close-paren before number:  )4   → )*4
          • close-paren before variable: )x  → )*x
          • close-paren before open-paren: )( → )*(

        Examples:
            ['2', 'x']          → ['2', '*', 'x']
            ['3', '(', 'x', '+', '2', ')'] stays — no implicit * needed
            ['2', '(', 'x', '+', '3', ')', '+', '1'] → same (( after operator is fine)
            ['3', '(', 'x', '+', '2', ')']  ← if '3' before '(' → ['3', '*', '(', ...]
        """
        result = []
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                curr_is_num_or_cparen = (tok.replace('.', '').replace('**', '').replace('-', '').replace('x', '1').isdigit()
                                         or tok == ')' or (tok[-1].isdigit() if tok else False))
                curr_is_num  = tok.replace('.', '').isdigit() or (tok[-1].isdigit() if tok else False)
                curr_is_cparen = tok == ')'
                nxt_is_var   = nxt.startswith('x') and (len(nxt) == 1 or nxt[1] in ('*', '+', '-', '/', ')'))
                nxt_is_oparen = nxt == '('
                nxt_is_num   = nxt.replace('.', '').isdigit()

                need_star = (
                    (curr_is_num   and nxt_is_var)    or   # 2x
                    (curr_is_num   and nxt_is_oparen) or   # 3(
                    (curr_is_cparen and nxt_is_num)   or   # )4
                    (curr_is_cparen and nxt_is_var)   or   # )x
                    (curr_is_cparen and nxt_is_oparen)     # )(
                )
                if need_star:
                    result.append('*')
        return result

    def handle_negatives(self, tokens: list[str]) -> list[str]:
        """
        Preserve negation: leading '-' or '-' immediately after an operator.

        Why: A minus sign at the start of an equation (or right after =, +, *)
        is a unary negation, not binary subtraction. Most parsers handle this
        correctly in string form, but we need to keep it as a token and not
        accidentally merge it with the following digit.

        Example:
            ['-', '5', '+', 'x', '=', '-', '3']
            → ['-5', '+', 'x', '=', '-3']  (merge into signed literals)
        """
        result = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == '-' and i + 1 < len(tokens):
                # Unary minus: at start OR after an operator or '='
                prev = result[-1] if result else None
                if prev is None or prev in self.OPERATOR_SYMBOLS:
                    # Merge with following number token
                    next_tok = tokens[i + 1]
                    if next_tok not in self.OPERATOR_SYMBOLS:
                        result.append('-' + next_tok)
                        i += 2
                        continue
            result.append(tok)
            i += 1
        return result

    def _build_string(self, tokens: list[str]) -> str:
        """
        Join tokens into a final equation string with appropriate spacing.

        Spaces around '+', '-', '=' improve readability.
        No spaces around '*' and '/' match conventional notation.
        """
        parts = []
        for tok in tokens:
            if tok in ('+', '-', '='):
                parts.append(f' {tok} ')
            else:
                parts.append(tok)
        return ''.join(parts).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def validate_equation(self, equation_str: str) -> tuple[bool, str, str]:
        """
        Validate and classify the equation string.

        Checks:
          • Exactly one '=' sign.
          • Content on both sides of '='.
          • No consecutive operators (++, --, */, etc.).
          • Balanced parentheses.
          • At least one numeric digit present.
          • SymPy can parse both sides without error.

        Returns:
            (is_valid, error_message, equation_type)
            equation_type is one of: 'arithmetic', 'linear', 'quadratic',
            'expression' (no '=' sign).
        """
        if not equation_str.strip():
            return False, 'Empty equation string.', 'unknown'

        # No '=' → expression to evaluate
        if '=' not in equation_str:
            return True, '', 'expression'

        # Multiple '=' signs
        if equation_str.count('=') > 1:
            return False, 'More than one equals sign found.', 'unknown'

        lhs_str, rhs_str = equation_str.split('=', 1)
        if not lhs_str.strip():
            return False, 'Nothing on left side of equals.', 'unknown'
        if not rhs_str.strip():
            return False, 'Nothing on right side of equals.', 'unknown'

        # Consecutive operators
        if re.search(r'[+\-*/]{2,}', equation_str.replace('**', '__')):
            return False, 'Consecutive operators detected.', 'unknown'

        # Balanced parentheses
        if equation_str.count('(') != equation_str.count(')'):
            return False, 'Unbalanced parentheses.', 'unknown'

        # Must contain a digit
        if not any(c.isdigit() for c in equation_str):
            return False, 'No numeric digits found.', 'unknown'

        # SymPy parse check
        if _SYMPY_OK:
            try:
                x = symbols('x')
                parse_expr(lhs_str.strip(), local_dict={'x': x})
                parse_expr(rhs_str.strip(), local_dict={'x': x})
            except Exception as e:
                return False, f'SymPy parse error: {e}', 'unknown'

        # Classify
        has_x    = 'x' in equation_str
        has_exp2 = 'x**2' in equation_str or 'x^2' in equation_str

        if not has_x:
            eq_type = 'arithmetic'
        elif has_exp2:
            eq_type = 'quadratic'
        else:
            eq_type = 'linear'

        return True, '', eq_type

    def to_sympy_format(self, equation_str: str) -> str:
        """
        Ensure the string uses Python/SymPy operator syntax.

        Converts: × → *,  ÷ → /,  ^ → **
        """
        return (equation_str
                .replace('×', '*')
                .replace('÷', '/')
                .replace('^',  '**'))

    def extract_variables(self, equation_str: str) -> list[str]:
        """
        Return a sorted list of unique variable names found in the equation.

        Example: '2*x + 3 = y' → ['x', 'y']
        """
        return sorted({m.group() for m in re.finditer(r'\b[a-zA-Z]\b', equation_str)
                       if m.group() not in ('e',)})  # exclude Euler's number

    # ─────────────────────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────────────────────

    def display_parse_steps(self, original_symbols: list,
                             final_equation: str) -> None:
        """
        Print each transformation step clearly for CV demonstration.

        Args:
            original_symbols: The raw symbol list (strings or dicts).
            final_equation:   The final equation string from parse().
        """
        print("\n" + "═" * 60)
        print("EQUATION PARSER — TRANSFORMATION STEPS")
        print("═" * 60)

        def _fmt(lst):
            return '[' + ',  '.join(f"'{t}'" for t in lst) + ']'

        raw = [s if isinstance(s, str) else s.get('symbol', '?')
               for s in original_symbols]

        print(f"\n  Step 1 — Raw recognised symbols:")
        print(f"           {_fmt(raw)}")

        if 'after_exponents' in self._steps:
            print(f"\n  Step 2 — After exponent merging (superscripts → **):")
            print(f"           {_fmt(self._steps['after_exponents'])}")

        if 'after_digit_merge' in self._steps:
            print(f"\n  Step 3 — After digit merging (consecutive digits → number):")
            print(f"           {_fmt(self._steps['after_digit_merge'])}")

        if 'after_implicit_mult' in self._steps:
            print(f"\n  Step 4 — After implicit multiplication insertion (2x → 2*x):")
            print(f"           {_fmt(self._steps['after_implicit_mult'])}")

        print(f"\n  Final equation string:")
        print(f"           \"{final_equation}\"")
        print("═" * 60 + "\n")
