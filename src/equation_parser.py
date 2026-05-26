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

    OPERATOR_SYMBOLS = {'+', '-', '*', '/', '='}
    DIGIT_SYMBOLS    = set('0123456789')

    def __init__(self):
        self._steps: dict = {}

    def parse(self, recognized_symbols) -> tuple[str, dict]:
        self._steps = {}

        if recognized_symbols and isinstance(recognized_symbols[0], dict):
            symbols_raw = [r['symbol']                       for r in recognized_symbols]
            types_raw   = [r.get('position_type', 'normal') for r in recognized_symbols]
        else:
            symbols_raw = list(recognized_symbols)
            types_raw   = ['normal'] * len(symbols_raw)

        self._steps['raw'] = symbols_raw.copy()

        tokens, types = self.handle_exponents(symbols_raw, types_raw)
        self._steps['after_exponents'] = tokens.copy()

        tokens = self.merge_digits(tokens)
        self._steps['after_digit_merge'] = tokens.copy()

        tokens = self.insert_implicit_multiply(tokens)
        self._steps['after_implicit_mult'] = tokens.copy()

        tokens = self.handle_negatives(tokens)
        self._steps['after_negatives'] = tokens.copy()

        equation_str = self._build_string(tokens)
        return equation_str, self._steps

    def handle_exponents(self, symbols: list[str],
                          types: list[str]) -> tuple[list[str], list[str]]:
        """Merge superscript digits into '**' notation, e.g. ['x', '2'(sup)] → ['x**2']."""
        result_syms  = []
        result_types = []
        i = 0
        while i < len(symbols):
            if (i + 1 < len(symbols)
                    and types[i + 1] == 'superscript'
                    and symbols[i + 1] in self.DIGIT_SYMBOLS
                    and symbols[i] not in self.OPERATOR_SYMBOLS):
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
        """Collapse consecutive single-digit tokens into multi-digit numbers."""
        result = []
        i = 0
        while i < len(tokens):
            if len(tokens[i]) == 1 and tokens[i] in self.DIGIT_SYMBOLS:
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
        """Insert '*' where handwriting convention implies multiplication (2x, 3(, )x, etc.)."""
        result = []
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                curr_is_num    = tok.replace('.', '').isdigit() or (tok[-1].isdigit() if tok else False)
                curr_is_cparen = tok == ')'
                nxt_is_var     = nxt.startswith('x') and (len(nxt) == 1 or nxt[1] in ('*', '+', '-', '/', ')'))
                nxt_is_oparen  = nxt == '('
                nxt_is_num     = nxt.replace('.', '').isdigit()

                need_star = (
                    (curr_is_num    and nxt_is_var)    or
                    (curr_is_num    and nxt_is_oparen) or
                    (curr_is_cparen and nxt_is_num)    or
                    (curr_is_cparen and nxt_is_var)    or
                    (curr_is_cparen and nxt_is_oparen)
                )
                if need_star:
                    result.append('*')
        return result

    def handle_negatives(self, tokens: list[str]) -> list[str]:
        """Merge unary minus with the following token to form signed literals."""
        result = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == '-' and i + 1 < len(tokens):
                prev = result[-1] if result else None
                if prev is None or prev in self.OPERATOR_SYMBOLS:
                    next_tok = tokens[i + 1]
                    if next_tok not in self.OPERATOR_SYMBOLS:
                        result.append('-' + next_tok)
                        i += 2
                        continue
            result.append(tok)
            i += 1
        return result

    def _build_string(self, tokens: list[str]) -> str:
        parts = []
        for tok in tokens:
            if tok in ('+', '-', '='):
                parts.append(f' {tok} ')
            else:
                parts.append(tok)
        return ''.join(parts).strip()

    def validate_equation(self, equation_str: str) -> tuple[bool, str, str]:
        if not equation_str.strip():
            return False, 'Empty equation string.', 'unknown'

        if '=' not in equation_str:
            return True, '', 'expression'

        if equation_str.count('=') > 1:
            return False, 'More than one equals sign found.', 'unknown'

        lhs_str, rhs_str = equation_str.split('=', 1)
        if not lhs_str.strip():
            return False, 'Nothing on left side of equals.', 'unknown'
        if not rhs_str.strip():
            return False, 'Nothing on right side of equals.', 'unknown'

        if re.search(r'[+\-*/]{2,}', equation_str.replace('**', '__')):
            return False, 'Consecutive operators detected.', 'unknown'

        if equation_str.count('(') != equation_str.count(')'):
            return False, 'Unbalanced parentheses.', 'unknown'

        if not any(c.isdigit() for c in equation_str):
            return False, 'No numeric digits found.', 'unknown'

        if _SYMPY_OK:
            try:
                x = symbols('x')
                parse_expr(lhs_str.strip(), local_dict={'x': x})
                parse_expr(rhs_str.strip(), local_dict={'x': x})
            except Exception as e:
                return False, f'SymPy parse error: {e}', 'unknown'

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
        return (equation_str
                .replace('×', '*')
                .replace('÷', '/')
                .replace('^',  '**'))

    def extract_variables(self, equation_str: str) -> list[str]:
        return sorted({m.group() for m in re.finditer(r'\b[a-zA-Z]\b', equation_str)
                       if m.group() not in ('e',)})

    def display_parse_steps(self, original_symbols: list, final_equation: str) -> None:
        print("\n" + "═" * 60)
        print("EQUATION PARSER — TRANSFORMATION STEPS")
        print("═" * 60)

        def _fmt(lst):
            return '[' + ',  '.join(f"'{t}'" for t in lst) + ']'

        raw = [s if isinstance(s, str) else s.get('symbol', '?')
               for s in original_symbols]

        print(f"\n  Raw symbols:          {_fmt(raw)}")

        if 'after_exponents' in self._steps:
            print(f"  After exponents:      {_fmt(self._steps['after_exponents'])}")
        if 'after_digit_merge' in self._steps:
            print(f"  After digit merge:    {_fmt(self._steps['after_digit_merge'])}")
        if 'after_implicit_mult' in self._steps:
            print(f"  After implicit mult:  {_fmt(self._steps['after_implicit_mult'])}")

        print(f"\n  Final: \"{final_equation}\"")
        print("═" * 60 + "\n")
