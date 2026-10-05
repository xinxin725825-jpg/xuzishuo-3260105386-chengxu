#!/usr/bin/env python3
"""A small, self-contained interpreter for the mini-Scheme specification."""

from __future__ import annotations

import operator
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Iterable


class SchemeError(Exception):
    """An error in a Scheme program."""


class Symbol(str):
    """A Scheme symbol, distinct from a string literal."""


class NilType:
    def __repr__(self) -> str:
        return "()"


NIL = NilType()


@dataclass
class Pair:
    """A Scheme pair (cons cell)."""

    car: Any
    cdr: Any


class Environment:
    def __init__(self, parent: Environment | None = None) -> None:
        self.values: dict[Symbol, Any] = {}
        self.parent = parent

    def define(self, name: Symbol, value: Any) -> None:
        self.values[name] = value

    def lookup(self, name: Symbol) -> Any:
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise SchemeError(f"unbound symbol: {name}")


@dataclass
class Builtin:
    name: str
    function: Callable[..., Any]


@dataclass
class Closure:
    parameters: list[Symbol]
    body: list[Any]
    environment: Environment


def tokenize(source: str) -> list[Any]:
    """Convert source into atoms and delimiter tokens, respecting strings/comments."""
    tokens: list[Any] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
        elif ch == ";":
            while i < len(source) and source[i] != "\n":
                i += 1
        elif ch in "()'":
            tokens.append(ch)
            i += 1
        elif ch == '"':
            i += 1
            chars: list[str] = []
            while i < len(source) and source[i] != '"':
                if source[i] == "\\":
                    i += 1
                    if i >= len(source):
                        raise SchemeError("unterminated string escape")
                    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                    chars.append(escapes.get(source[i], "\\" + source[i]))
                else:
                    chars.append(source[i])
                i += 1
            if i >= len(source):
                raise SchemeError("unterminated string")
            i += 1
            tokens.append(("STRING", "".join(chars)))
        else:
            start = i
            while i < len(source) and not source[i].isspace() and source[i] not in "();'\"":
                i += 1
            atom = source[start:i]
            if atom == "#t":
                tokens.append(True)
            elif atom == "#f":
                tokens.append(False)
            else:
                try:
                    tokens.append(int(atom, 10))
                except ValueError:
                    try:
                        # Do not treat ordinary symbols containing dots as floats.
                        if any(c in atom for c in ".eE") and atom not in (".",):
                            tokens.append(float(atom))
                        else:
                            tokens.append(Symbol(atom))
                    except ValueError:
                        tokens.append(Symbol(atom))
    return tokens


def parse(tokens: list[Any]) -> list[Any]:
    """Read all top-level forms from tokens."""
    position = 0

    def one() -> Any:
        nonlocal position
        if position >= len(tokens):
            raise SchemeError("unexpected end of input")
        token = tokens[position]
        position += 1
        if token == "(":
            items: list[Any] = []
            while position < len(tokens) and tokens[position] != ")":
                if tokens[position] == ".":
                    position += 1
                    if not items:
                        raise SchemeError("dot cannot begin a list")
                    tail = one()
                    if position >= len(tokens) or tokens[position] != ")":
                        raise SchemeError("dotted list must have one tail")
                    position += 1
                    return build_list(items, tail)
                items.append(one())
            if position >= len(tokens):
                raise SchemeError("unclosed list")
            position += 1
            return items
        if token == ")":
            raise SchemeError("unexpected closing parenthesis")
        if token == "'":
            return [Symbol("quote"), one()]
        if isinstance(token, tuple) and token[0] == "STRING":
            return token[1]
        return token

    expressions: list[Any] = []
    while position < len(tokens):
        expressions.append(one())
    return expressions


def build_list(items: Iterable[Any], tail: Any = NIL) -> Any:
    result = tail
    for item in reversed(list(items)):
        result = Pair(item, result)
    return result


def quote_datum(value: Any) -> Any:
    if isinstance(value, list):
        return build_list(quote_datum(item) for item in value)
    return value


def is_false(value: Any) -> bool:
    return value is False


def sequence(expressions: list[Any], env: Environment) -> Any:
    result: Any = None
    for expression in expressions:
        result = evaluate(expression, env)
    return result


def evaluate(expression: Any, env: Environment) -> Any:
    if isinstance(expression, Symbol):
        return env.lookup(expression)
    if not isinstance(expression, list):
        return expression
    if not expression:
        return NIL

    head = expression[0]
    if isinstance(head, Symbol):
        name = str(head)
        args = expression[1:]
        if name == "quote":
            if len(args) != 1:
                raise SchemeError("quote expects one argument")
            return quote_datum(args[0])
        if name == "if":
            if len(args) not in (2, 3):
                raise SchemeError("if expects two or three arguments")
            test = evaluate(args[0], env)
            if not is_false(test):
                return evaluate(args[1], env)
            return evaluate(args[2], env) if len(args) == 3 else None
        if name == "cond":
            for clause in args:
                if not isinstance(clause, list) or not clause:
                    raise SchemeError("invalid cond clause")
                test_expr, *body = clause
                if isinstance(test_expr, Symbol) and test_expr == "else":
                    return sequence(body, env)
                test = evaluate(test_expr, env)
                if not is_false(test):
                    return sequence(body, env) if body else test
            return None
        if name == "and":
            result: Any = True
            for arg in args:
                result = evaluate(arg, env)
                if is_false(result):
                    return False
            return result
        if name == "or":
            for arg in args:
                result = evaluate(arg, env)
                if not is_false(result):
                    return result
            return False
        if name == "define":
            if len(args) < 2:
                raise SchemeError("define expects a name and value")
            target = args[0]
            if isinstance(target, list) and target and isinstance(target[0], Symbol):
                name_sym = target[0]
                closure = Closure(target[1:], args[1:], env)
                env.define(name_sym, closure)
                return name_sym
            if not isinstance(target, Symbol) or len(args) != 2:
                raise SchemeError("invalid define")
            value = evaluate(args[1], env)
            env.define(target, value)
            return target
        if name == "lambda":
            if len(args) < 2 or not isinstance(args[0], list):
                raise SchemeError("lambda expects parameters and a body")
            params = args[0]
            if not all(isinstance(param, Symbol) for param in params):
                raise SchemeError("lambda parameters must be symbols")
            return Closure(params, args[1:], env)
        if name == "let":
            if len(args) < 2 or not isinstance(args[0], list):
                raise SchemeError("let expects bindings and a body")
            # Scheme let initializers all use the outer environment.
            names: list[Symbol] = []
            values: list[Any] = []
            for binding in args[0]:
                if not isinstance(binding, list) or len(binding) != 2 or not isinstance(binding[0], Symbol):
                    raise SchemeError("invalid let binding")
                names.append(binding[0])
                values.append(evaluate(binding[1], env))
            local = Environment(env)
            for key, value in zip(names, values):
                local.define(key, value)
            return sequence(args[1:], local)
        if name == "begin":
            return sequence(args, env)

    procedure = evaluate(head, env)
    values = [evaluate(arg, env) for arg in expression[1:]]
    return apply(procedure, values)


def apply(procedure: Any, args: list[Any]) -> Any:
    if isinstance(procedure, Builtin):
        try:
            return procedure.function(*args)
        except SchemeError:
            raise
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise SchemeError(f"{procedure.name}: {exc}") from exc
    if isinstance(procedure, Closure):
        if len(args) != len(procedure.parameters):
            raise SchemeError("wrong number of arguments")
        local = Environment(procedure.environment)
        for name, value in zip(procedure.parameters, args):
            local.define(name, value)
        return sequence(procedure.body, local)
    raise SchemeError(f"not a procedure: {write(procedure)}")


def pair_to_python_list(value: Any) -> tuple[list[Any], Any]:
    values: list[Any] = []
    seen: set[int] = set()
    while isinstance(value, Pair):
        if id(value) in seen:
            raise SchemeError("cyclic list")
        seen.add(id(value))
        values.append(value.car)
        value = value.cdr
    return values, value


def require_list(value: Any, proper: bool = True) -> list[Any]:
    if value is NIL:
        return []
    if not isinstance(value, Pair):
        raise SchemeError("expected a list")
    items, tail = pair_to_python_list(value)
    if proper and tail is not NIL:
        raise SchemeError("expected a proper list")
    return items


def scheme_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Pair):
        return scheme_equal(left.car, right.car) and scheme_equal(left.cdr, right.cdr)
    if left is NIL:
        return True
    if isinstance(left, (Closure, Builtin)):
        return left is right
    return left == right


def scheme_eq(left: Any, right: Any) -> bool:
    if isinstance(left, (Pair, Closure, Builtin)) or left is NIL or right is NIL:
        return left is right
    if type(left) is not type(right):
        return False
    return left == right


def require_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemeError("expected a number")
    return value


def trunc_div(a: int, b: int) -> int:
    return (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)


def arithmetic() -> dict[str, Callable[..., Any]]:
    def add(*xs: Any) -> Any:
        return sum(require_number(x) for x in xs)

    def multiply(*xs: Any) -> Any:
        return reduce(operator.mul, (require_number(x) for x in xs), 1)

    def subtract(*xs: Any) -> Any:
        nums = [require_number(x) for x in xs]
        if not nums:
            raise SchemeError("- expects at least one argument")
        return -nums[0] if len(nums) == 1 else reduce(operator.sub, nums[1:], nums[0])

    def divide(*xs: Any) -> Any:
        nums = [require_number(x) for x in xs]
        if not nums:
            raise SchemeError("/ expects at least one argument")
        if len(nums) == 1:
            return 1 / nums[0]
        result = nums[0]
        for value in nums[1:]:
            result = result / value if isinstance(result, float) or isinstance(value, float) else trunc_div(result, value)
        return result

    def modulo(a: Any, b: Any) -> int:
        a, b = require_number(a), require_number(b)
        return int(a) % int(b)

    def quotient(a: Any, b: Any) -> int:
        return trunc_div(int(require_number(a)), int(require_number(b)))

    def expt(a: Any, b: Any) -> Any:
        return require_number(a) ** require_number(b)

    def compare(fn: Callable[[Any, Any], bool]) -> Callable[..., bool]:
        def inner(*xs: Any) -> bool:
            if len(xs) < 2:
                return True
            for value in xs:
                if isinstance(value, bool) or not isinstance(value, (int, float, Symbol, str)):
                    raise SchemeError("comparison expects numbers or symbols")
            return all(fn(a, b) for a, b in zip(xs, xs[1:]))
        return inner

    return {
        "+": add, "-": subtract, "*": multiply, "/": divide,
        "modulo": modulo, "quotient": quotient, "expt": expt,
        "abs": lambda x: abs(require_number(x)),
        "=": compare(operator.eq), "<": compare(operator.lt), ">": compare(operator.gt),
        "<=": compare(operator.le), ">=": compare(operator.ge),
    }


def initial_environment() -> Environment:
    env = Environment()

    def add_builtin(name: str, function: Callable[..., Any]) -> None:
        env.define(Symbol(name), Builtin(name, function))

    for name, function in arithmetic().items():
        add_builtin(name, function)
    add_builtin("not", lambda x: is_false(x))
    add_builtin("cons", lambda a, b: Pair(a, b))
    add_builtin("car", lambda x: x.car if isinstance(x, Pair) else _bad_pair("car"))
    add_builtin("cdr", lambda x: x.cdr if isinstance(x, Pair) else _bad_pair("cdr"))
    add_builtin("list", lambda *xs: build_list(xs))
    add_builtin("length", lambda x: len(require_list(x)))
    add_builtin("append", lambda *xs: _append(xs))
    add_builtin("null?", lambda x: x is NIL)
    add_builtin("pair?", lambda x: isinstance(x, Pair))
    add_builtin("list?", lambda x: _is_proper_list(x))
    add_builtin("number?", lambda x: isinstance(x, (int, float)) and not isinstance(x, bool))
    add_builtin("boolean?", lambda x: isinstance(x, bool))
    add_builtin("symbol?", lambda x: isinstance(x, Symbol))
    add_builtin("string?", lambda x: isinstance(x, str) and not isinstance(x, Symbol))
    add_builtin("procedure?", lambda x: isinstance(x, (Builtin, Closure)))
    add_builtin("zero?", lambda x: require_number(x) == 0)
    add_builtin("even?", lambda x: int(require_number(x)) % 2 == 0)
    add_builtin("odd?", lambda x: int(require_number(x)) % 2 != 0)
    add_builtin("eq?", scheme_eq)
    add_builtin("equal?", scheme_equal)
    add_builtin("display", _display)
    add_builtin("newline", _newline)
    return env


def _bad_pair(name: str) -> Any:
    raise SchemeError(f"{name} expects a pair")


def _is_proper_list(value: Any) -> bool:
    if value is NIL:
        return True
    if not isinstance(value, Pair):
        return False
    _, tail = pair_to_python_list(value)
    return tail is NIL


def _append(values: tuple[Any, ...]) -> Any:
    if not values:
        return NIL
    items: list[Any] = []
    for value in values[:-1]:
        items.extend(require_list(value))
    last = values[-1]
    items.extend(require_list(last))
    return build_list(items)


def _display(value: Any) -> None:
    print(value if isinstance(value, str) and not isinstance(value, Symbol) else write(value, display=True), end="")
    return None


def _newline() -> None:
    print()
    return None


def write(value: Any, display: bool = False) -> str:
    if value is None:
        return ""
    if value is True:
        return "#t"
    if value is False:
        return "#f"
    if value is NIL:
        return "()"
    if isinstance(value, Symbol):
        return str(value)
    if isinstance(value, str):
        if display:
            return value
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, Pair):
        items, tail = pair_to_python_list(value)
        prefix = " ".join(write(item) for item in items)
        if tail is NIL:
            return f"({prefix})"
        return f"({prefix} . {write(tail)})"
    if isinstance(value, (Closure, Builtin)):
        return "#<procedure>"
    if isinstance(value, list):
        return "(" + " ".join(write(item) for item in value) + ")"
    return str(value)


def run(source: str, env: Environment) -> None:
    for expression in parse(tokenize(source)):
        result = evaluate(expression, env)
        if result is not None:
            print(write(result))


def main(argv: list[str]) -> int:
    env = initial_environment()
    if not argv:
        source = sys.stdin.read()
        run(source, env)
        return 0
    for filename in argv:
        try:
            source = Path(filename).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cannot read {filename}: {exc}", file=sys.stderr)
            return 1
        run(source, env)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SchemeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
