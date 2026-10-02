#!/usr/bin/env python3
"""
Datalog engine — rule-based inference backend for the pyke tool.

Reads a standard Datalog program file, performs forward-chaining inference,
and prints query results.

Usage:
    python3 datalog_engine.py <program.dl>

Syntax (what the LLM generates):
    % comment
    predicate(constant1, constant2, ...).                ← fact
    head(Var1, Var2) :- body1(Var1), body2(Var2).        ← rule
    ?- query_predicate(term1, term2).                    ← query
    ?- not query_predicate(term1, term2).                ← negated query

Terminology rules:
    - Constants start with a lowercase letter:  bear, bald_eagle
    - Variables start with an uppercase letter: X, Y, Name
    - Predicates start with a lowercase letter

Output (stdout):
    One result per query line: "TRUE: <query>" or "FALSE: <query>".

    FALSE means the fact cannot be derived from the database
    (Open World Assumption).

Exit codes:
    0 — success (even when individual queries return FALSE)
    1 — syntax error
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List, Set, Tuple, Optional


# ---------------------------------------------------------------------------
# Tokenizer / Parser
# ---------------------------------------------------------------------------

TOKEN_SPLIT = re.compile(r"""
    (?:'[^']*')|
    (?:"[^"]*")|
    (?:[A-Z][a-zA-Z0-9_]*)|
    (?:[a-z][a-zA-Z0-9_]*)|
    (?:[0-9]+)|
    \?-|
    :-|
    [().,;?\-]
""", re.VERBOSE)


class ParseError(Exception):
    """Raised when the Datalog source cannot be parsed."""


def tokenize(line: str) -> List[str]:
    """Split a Datalog source line into tokens."""
    # Strip everything after an unquoted % (comment)
    comment_idx = line.find("%")
    if comment_idx >= 0:
        line = line[:comment_idx]

    tokens = TOKEN_SPLIT.findall(line)
    return [t for t in tokens if t.strip()]


def parse_fact(tokens: List[str]) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """
    Parse a fact: predicate(t1, t2, ...).
    Returns (pred_name, (t1, t2, ...)) or None.
    """
    if len(tokens) < 3:
        return None
    pred = tokens[0]
    if not pred[0].islower():
        return None
    if tokens[1] != "(":
        return None

    try:
        close_idx = tokens.index(")", 2)
    except ValueError:
        raise ParseError(
            f"Missing closing ')' in fact: {' '.join(tokens)}"
        )

    args = []
    for t in tokens[2:close_idx]:
        if t == ",":
            continue
        args.append(t)

    return (pred, tuple(args))


def parse_rule(
    tokens: List[str],
) -> Optional[Tuple[str, Tuple[str, ...], List[Tuple[str, Tuple[str, ...]]]]]:
    """
    Parse a rule: head(t1, t2) :- body1(x), body2(y).
    Returns (head_pred, head_args, [(body_pred1, body_args1), ...]) or None.
    """
    if ":-" not in tokens:
        return None

    if_idx = tokens.index(":-")

    # Parse head
    head_tokens = tokens[:if_idx]
    head = parse_fact(head_tokens + ["."])
    if head is None:
        return None
    head_pred, head_args = head

    # Parse body — use parenthesis-depth splitting for comma-separated literals
    body_tokens = tokens[if_idx + 1:]
    if body_tokens and body_tokens[-1] == ".":
        body_tokens = body_tokens[:-1]

    literals = _split_body_literals(body_tokens)
    body: list = []
    for lit_tokens in literals:
        lit = parse_fact(lit_tokens)
        if lit:
            body.append(lit)

    if not body:
        raise ParseError(f"Empty rule body: {' '.join(tokens)}")

    return (head_pred, head_args, body)


def parse_query(
    tokens: List[str],
) -> Optional[Tuple[bool, str, Tuple[str, ...]]]:
    """
    Parse a query: ?- pred(t1, t2).  or  ?- not pred(t1, t2).
    Returns (negated: bool, pred_name, args) or None.
    """
    if not tokens or tokens[0] != "?-":
        return None

    rest = tokens[1:]
    negated = False
    if rest and rest[0] in ("not", "-"):
        negated = True
        rest = rest[1:]

    if len(rest) < 3:
        raise ParseError(f"Invalid query: {' '.join(tokens)}")

    pred = rest[0]
    if rest[1] != "(":
        raise ParseError(f"Expected '(' in query: {' '.join(tokens)}")

    try:
        close_idx = rest.index(")", 2)
    except ValueError:
        raise ParseError(f"Missing ')' in query")

    args = []
    for t in rest[2:close_idx]:
        if t == ",":
            continue
        args.append(t)

    return (negated, pred, tuple(args))


# ---------------------------------------------------------------------------
# Helper — split body tokens at commas, respecting parenthesis depth
# ---------------------------------------------------------------------------

def _split_body_literals(tokens: List[str]) -> List[List[str]]:
    """Split tokens on commas that appear at depth 0 in parentheses."""
    literals: list = []
    current: list = []
    depth = 0
    for t in tokens:
        if t == "," and depth == 0:
            if current:
                literals.append(current)
                current = []
        else:
            if t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
            current.append(t)
    if current:
        literals.append(current)
    return literals


# ---------------------------------------------------------------------------
# Datalog Engine — Naive Forward Chaining
# ---------------------------------------------------------------------------

class DatalogProgram:
    """A parsed Datalog program with facts, rules, and queries."""

    def __init__(self):
        self.facts: List[Tuple[str, Tuple[str, ...]]] = []
        self.rules: List[
            Tuple[str, Tuple[str, ...], List[Tuple[str, Tuple[str, ...]]]]
        ] = []
        self.queries: List[Tuple[bool, str, Tuple[str, ...]]] = []

    def parse_file(self, filepath: str) -> None:
        """Parse a Datalog program from a file."""
        with open(filepath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("%"):
                    continue

                try:
                    tokens = tokenize(line)
                    if not tokens:
                        continue

                    # Dispatch by leading token
                    if tokens[0] == "?-":
                        q = parse_query(tokens)
                        if q:
                            self.queries.append(q)
                        else:
                            raise ParseError(
                                f"Cannot parse query at line {lineno}: {line}"
                            )
                    elif ":-" in tokens:
                        r = parse_rule(tokens)
                        if r:
                            self.rules.append(r)
                        else:
                            raise ParseError(
                                f"Cannot parse rule at line {lineno}: {line}"
                            )
                    else:
                        fct = parse_fact(tokens)
                        if fct:
                            self.facts.append(fct)
                        else:
                            raise ParseError(
                                f"Cannot parse line {lineno}: {line}"
                            )

                except ParseError as e:
                    print(f"ParseError: {e}", file=sys.stderr)
                    sys.exit(1)

    def evaluate(self) -> Dict[str, Set[Tuple[str, ...]]]:
        """
        Naive forward-chaining evaluation.

        1. Start from EDB (facts).
        2. Repeatedly apply rules to derive new facts.
        3. Stop at fixpoint (no new facts).

        Returns:
            Database mapping predicate_name → set of fact tuples.
        """
        # db: predicate_name → set of fact tuples
        db: Dict[str, Set[Tuple[str, ...]]] = {}

        # Load facts (EDB)
        for pred, args in self.facts:
            db.setdefault(pred, set()).add(args)

        # Forward chaining
        changed = True
        iteration = 0
        max_iterations = 10_000  # safety ceiling

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for head_pred, head_args, body in self.rules:
                all_bindings = self._evaluate_body(body, db)

                for binding in all_bindings:
                    # Substitute variables in the head
                    instantiated_head = tuple(
                        binding.get(arg, arg) for arg in head_args
                    )
                    # Safety: every variable in the head must be bound
                    if any(arg[0].isupper() for arg in instantiated_head):
                        continue

                    if instantiated_head not in db.get(head_pred, set()):
                        db.setdefault(head_pred, set()).add(instantiated_head)
                        changed = True

        if iteration >= max_iterations:
            print(
                "Warning: max iterations reached — possible infinite recursion.",
                file=sys.stderr,
            )

        return db

    def _evaluate_body(
        self,
        body: List[Tuple[str, Tuple[str, ...]]],
        db: Dict[str, Set[Tuple[str, ...]]],
    ) -> List[Dict[str, str]]:
        """Evaluate a rule body and return all satisfying variable bindings."""
        if not body:
            return [{}]

        bindings: list = [{}]

        for pred, args in body:
            if pred not in db:
                return []  # predicate has no facts → body cannot be satisfied
            new_bindings: list = []
            for fact in db[pred]:
                for b in bindings:
                    merged = self._unify(args, fact, dict(b))
                    if merged is not None:
                        new_bindings.append(merged)
            bindings = new_bindings
            if not bindings:
                return []

        return bindings

    @staticmethod
    def _unify(
        pattern: Tuple[str, ...],
        fact: Tuple[str, ...],
        binding: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        """
        Attempt to unify *pattern* with *fact* under *binding*.

        Returns a new (extended) binding on success, or None on failure.
        """
        if len(pattern) != len(fact):
            return None

        new_binding = dict(binding)
        for p, f in zip(pattern, fact):
            if p[0].isupper():  # variable
                if p in new_binding:
                    if new_binding[p] != f:
                        return None
                else:
                    new_binding[p] = f
            else:  # constant
                if p != f:
                    return None
        return new_binding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <program.dl>", file=sys.stderr)
        sys.exit(1)

    program = DatalogProgram()

    try:
        program.parse_file(sys.argv[1])
    except ParseError as e:
        print(f"ParseError: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file not found — {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    if not program.queries:
        print("Error: no queries found in program.", file=sys.stderr)
        sys.exit(1)

    db = program.evaluate()

    for negated, pred, args in program.queries:
        is_derivable = (pred in db and args in db[pred])

        if negated:
            result = "FALSE" if is_derivable else "TRUE"
        else:
            result = "TRUE" if is_derivable else "FALSE"

        query_str = f"{pred}({','.join(args)})"
        if negated:
            query_str = f"not {query_str}"
        print(f"{result}: {query_str}")

    sys.exit(0)


if __name__ == "__main__":
    main()
