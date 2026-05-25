"""
Stage 4 - Module 2: Step Formatter

Transforms the raw solution dictionary from MathSolver into polished,
presentation-ready output for the terminal, the web application, and
LaTeX-rendered math displays.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sympy import sympify, latex as sympy_latex, parse_expr, symbols, SympifyError
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

# Mathematical property names used in step explanations
PROPERTY_NAMES = {
    'subtraction':    'Subtraction Property of Equality',
    'addition':       'Addition Property of Equality',
    'division':       'Division Property of Equality',
    'multiplication': 'Multiplication Property of Equality',
    'zero_product':   'Zero Product Property',
    'quadratic':      'Quadratic Formula',
    'substitution':   'Substitution Principle',
    'distributive':   'Distributive Property',
}


class StepFormatter:
    """
    Formats MathSolver result dictionaries for display in the terminal
    (box-drawing characters) or in the web app (LaTeX / Markdown).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Terminal display
    # ─────────────────────────────────────────────────────────────────────────

    def format_for_display(self, solution_dict: dict) -> str:
        """
        Create a professionally bordered text representation of the solution.

        Uses Unicode box-drawing characters for a clean terminal display.
        Structure:
            ┌─ Header: equation and type ─┐
            │ Numbered steps              │
            ├─ Answer ────────────────────┤
            │ Verification               │
            └─────────────────────────────┘

        Args:
            solution_dict: Standardised result dict from MathSolver.

        Returns:
            Multi-line formatted string ready for print().
        """
        if not solution_dict.get('success', False):
            err = solution_dict.get('error', 'Unknown error')
            return (
                "╔══════════════════════════════════════╗\n"
                f"║  ERROR: {err:<30}║\n"
                "╚══════════════════════════════════════╝"
            )

        W = 68   # Box width (inner)
        SEP = '─' * W

        def box_line(text='', pad=1):
            return f"│{'':>{pad}}{text:<{W - pad}}│"

        lines = []
        lines.append('┌' + SEP + '┐')
        lines.append(box_line())
        lines.append(box_line(f"  Equation : {solution_dict.get('original_equation', '')}"))
        lines.append(box_line(f"  Type     : {solution_dict.get('equation_type', '').capitalize()}"))
        lines.append(box_line())
        lines.append('├' + SEP + '┤')

        # Steps
        for step in solution_dict.get('steps', []):
            num  = step.get('step_number', '?')
            desc = step.get('description', '')
            expr = step.get('expression', '')
            expl = step.get('explanation', '')

            lines.append(box_line())
            lines.append(box_line(f"  Step {num}: {desc}"))
            lines.append(box_line(f"    → {expr}"))
            if expl:
                # Word-wrap explanation at ~60 chars
                for chunk in [expl[i:i+60] for i in range(0, len(expl), 60)]:
                    lines.append(box_line(f"       {chunk}"))

        lines.append(box_line())
        lines.append('├' + SEP + '┤')

        # Answer
        answer = solution_dict.get('answer_str', '')
        lines.append(box_line())
        lines.append(box_line(f"  ★  ANSWER:  {answer}"))
        lines.append(box_line())

        # Verification
        ver = solution_dict.get('verification', {})
        if ver.get('verified') is not None:
            status = '✓ Verified correct' if ver['verified'] else '✗ Verification failed'
            lines.append(box_line(f"  {status}"))
            if 'left_value' in ver and 'right_value' in ver:
                lines.append(box_line(
                    f"  Substitution: LHS={ver['left_value']}  "
                    f"RHS={ver['right_value']}"
                ))

        meta = solution_dict.get('metadata', {})
        lines.append(box_line())
        lines.append(box_line(
            f"  Steps: {meta.get('num_steps', '?')}   "
            f"Solved in {meta.get('solve_time_ms', 0):.1f} ms"
        ))
        lines.append(box_line())
        lines.append('└' + SEP + '┘')

        return '\n'.join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # LaTeX output
    # ─────────────────────────────────────────────────────────────────────────

    def format_for_latex(self, solution_dict: dict) -> dict:
        """
        Convert all mathematical expressions in every step to LaTeX strings.

        Uses sympy.latex() so expressions are correctly typeset when rendered
        in the web application with MathJax or KaTeX.

        Args:
            solution_dict: Standardised result dict.

        Returns:
            New dict identical to solution_dict but with 'expression' fields
            replaced by LaTeX strings and an added 'latex_answer' key.
        """
        import copy
        result = copy.deepcopy(solution_dict)

        if _SYMPY_OK:
            x = symbols('x')
            local = {'x': x}

            def _to_latex(expr_str: str) -> str:
                """Best-effort conversion of expr_str to LaTeX."""
                try:
                    # Try as an equation first
                    if '=' in expr_str and '==' not in expr_str:
                        parts = expr_str.split('=', 1)
                        lhs_l = sympy_latex(parse_expr(parts[0].strip(), local_dict=local))
                        rhs_l = sympy_latex(parse_expr(parts[1].strip(), local_dict=local))
                        return f'{lhs_l} = {rhs_l}'
                    return sympy_latex(parse_expr(expr_str, local_dict=local))
                except Exception:
                    return expr_str   # Fall back to plain string

            for step in result.get('steps', []):
                step['expression_latex'] = _to_latex(step.get('expression', ''))

            # LaTeX for the final answer
            raw_ans = result.get('answer_str', '')
            result['latex_answer'] = _to_latex(raw_ans) if raw_ans else ''

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Step enrichment
    # ─────────────────────────────────────────────────────────────────────────

    def format_step_explanation(self, step: dict) -> dict:
        """
        Enrich a single step dict with formal property name and teaching note.

        Adds:
            property_name  – formal math property name (if identifiable)
            teaching_note  – plain-English 'why this step matters' sentence
            concept        – the underlying algebraic concept

        Args:
            step: Step dict from MathSolver.

        Returns:
            Enriched step dict (new dict, original unchanged).
        """
        import copy
        enriched = copy.copy(step)
        desc = (step.get('description', '') + ' ' +
                step.get('explanation', '')).lower()

        # Identify the property
        if 'subtraction property' in desc or 'subtract' in desc:
            enriched['property_name'] = PROPERTY_NAMES['subtraction']
            enriched['teaching_note'] = (
                "Subtracting the same value from both sides maintains equality. "
                "This is how we move constants away from the variable."
            )
            enriched['concept'] = 'Balancing equations'

        elif 'addition property' in desc or 'add' in desc:
            enriched['property_name'] = PROPERTY_NAMES['addition']
            enriched['teaching_note'] = (
                "Adding the same value to both sides maintains equality."
            )
            enriched['concept'] = 'Balancing equations'

        elif 'division property' in desc or 'divid' in desc:
            enriched['property_name'] = PROPERTY_NAMES['division']
            enriched['teaching_note'] = (
                "Dividing both sides by the coefficient isolates the variable. "
                "This is the final step to solve for x."
            )
            enriched['concept'] = 'Isolating the variable'

        elif 'multiplication property' in desc or 'multipl' in desc:
            enriched['property_name'] = PROPERTY_NAMES['multiplication']
            enriched['teaching_note'] = (
                "Multiplying both sides by the same value maintains equality."
            )
            enriched['concept'] = 'Isolating the variable'

        elif 'zero product' in desc:
            enriched['property_name'] = PROPERTY_NAMES['zero_product']
            enriched['teaching_note'] = (
                "If a product of factors equals zero, at least one factor must "
                "be zero. This converts a quadratic into two linear equations."
            )
            enriched['concept'] = 'Zero Product Property'

        elif 'quadratic formula' in desc:
            enriched['property_name'] = PROPERTY_NAMES['quadratic']
            enriched['teaching_note'] = (
                "The quadratic formula always works, even when factoring fails."
            )
            enriched['concept'] = 'Quadratic Formula'

        elif 'verif' in desc or 'substitut' in desc:
            enriched['property_name'] = PROPERTY_NAMES['substitution']
            enriched['teaching_note'] = (
                "Substituting our answer back into the original equation is the "
                "only reliable way to confirm the solution is correct."
            )
            enriched['concept'] = 'Answer verification'

        else:
            enriched.setdefault('property_name', '')
            enriched.setdefault('teaching_note', step.get('explanation', ''))
            enriched.setdefault('concept', '')

        return enriched

    # ─────────────────────────────────────────────────────────────────────────
    # Quick summary
    # ─────────────────────────────────────────────────────────────────────────

    def short_summary(self, solution_dict: dict) -> str:
        """
        One-line summary suitable for the web app answer box.

        Returns:
            e.g. 'x = 5' or 'x = -2, x = -3' or 'Error: ...'
        """
        if not solution_dict.get('success', False):
            return f"Error: {solution_dict.get('error', 'Unknown')}"
        return solution_dict.get('answer_str', 'No answer')
