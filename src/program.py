import inspect
from typing import Callable, TypeVar, Generic, ParamSpec, Union, Any, Optional
from types import EllipsisType
import re

from .errors import (
    NotCallableError,
    TooFewArgumentsError,
    InvalidArgumentTypeError,
    VariableEvalError,
)

from copy import deepcopy

R = TypeVar("R")
P = ParamSpec("P")


class Program(Generic[R]):
    """
    Represents a program that can be partially applied and composed.

    Programs can contain holes (Variables) that are filled through partial application.
    Uses De Bruijn indexing for variable management.
    """

    def __init__(self, func: Callable[..., R], cost: int = 1):
        """
        Constructs a Program from a Callable(python function).
        """
        if not callable(func):
            raise NotCallableError(func)

        self._func: Callable[..., R] = func
        self.__name__ = f"<Program: {getattr(func, '__name__')}>"
        self._required_args = len(
            [
                param
                for param in inspect.signature(func).parameters.values()
                if param.default is inspect.Parameter.empty
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        )
        self._args: list[Program] = []
        self._cost = cost

    @property
    def arity(self) -> int:
        """Number of arguments this program expects."""
        return self._required_args

    def __call__(self, *args: "Union[Program, EllipsisType]") -> "Program[R]":
        """
        Compose this program with arguments, creating holes for ellipsis.

        Args:
            *args: Programs or ellipsis (...) to create holes

        Returns:
            New composed program
        """
        processed_args = []
        for arg in args:
            if arg is Ellipsis:
                # Placeholder index, will be renumbered
                processed_args.append(Variable(-1))
            elif isinstance(arg, Program):
                processed_args.append(arg)
            else:
                raise InvalidArgumentTypeError()
        if len(args) < self.arity:
            raise TooFewArgumentsError(self.__name__, self.arity, len(args))

        composed = Program(self._func)
        composed._args = processed_args
        composed.__name__ = self.__name__
        composed._renumber_variables()
        return composed

    def __deepcopy__(self, memo: dict[int, Any]) -> "Program[R]":
        """Create a deep copy of this program."""
        if isinstance(self, Variable):
            return Variable(self.index)

        cloned = Program(self._func)
        cloned._args = [deepcopy(arg, memo) for arg in self._args]
        cloned.__name__ = self.__name__
        return cloned

    def apply(self, *primitives: "Union[Program, EllipsisType]") -> "Program[R]":
        """
        Apply primitives to variables (holes) in the program.

        Args:
            *primitives: Values to substitute for variables, in De Bruijn index order

        Returns:
            New program with substitutions applied
        """
        variables = sorted(self._collect_variables())
        self._validate_primitive_count(primitives, variables)

        substitution_map = self._create_substitution_map(primitives, variables)
        result = self._substitute_variables(substitution_map)

        result._renumber_variables()
        result._required_args = len(result._collect_variables())
        return result

    def __repr__(
        self, indent: str = "", is_last: bool = True, is_root: bool = True
    ) -> str:
        """Generate a tree representation of the program structure."""
        if is_root:
            lines = [self.__name__]
            child_indent = ""
        else:
            connector = "└── " if is_last else "├── "
            lines = [f"{indent}{connector}{self.__name__}"]
            child_indent = indent + ("    " if is_last else "│   ")

        for i, arg in enumerate(self._args):
            is_last_child = i == len(self._args) - 1
            lines.append(arg.__repr__(child_indent, is_last_child, is_root=False))

        return "\n".join(lines)

    def __str__(self) -> str:
        """Generate a Lisp-like string representation with optional (lam ...) if variables are free."""

        def stringify(prog: Program) -> str:
            if isinstance(prog, Variable):
                return str(prog)
            if not prog._args:
                return prog.__name__.split(":")[-1].strip(" >")
            func_name = prog.__name__.split(":")[-1].strip(" >").lower()
            args_str = " ".join(stringify(arg) for arg in prog._args)
            return f"({func_name} {args_str})"

        body = stringify(self)
        if self._collect_variables():
            return f"(lam {body})"
        return body

    @classmethod
    def parse(cls, s: str, env: dict[str, "Program"]):
        """
        Reconstructs a Program object from its string representation.

        Args:
            s: The string (from __str__) to parse.
            env: A mapping from function names to Program instances.

        Returns:
            A reconstructed Program object.
        """
        tokens = re.findall(r"\(|\)|[^\s()]+", s)
        ast = cls._parse_tokens(tokens)
        return cls._build_from_ast(ast, env)

    @classmethod
    def _parse_tokens(cls, tokens: list[str]) -> Union[str, list]:
        if not tokens:
            raise SyntaxError("Unexpected EOF while reading")
        token = tokens.pop(0)
        if token == "(":
            result = []
            while tokens and tokens[0] != ")":
                result.append(cls._parse_tokens(tokens))
            if not tokens:
                raise SyntaxError("Unmatched '('")
            tokens.pop(0)  # Remove ')'
            return result
        elif token == ")":
            raise SyntaxError("Unexpected ')'")
        else:
            return token

    @classmethod
    def _build_from_ast(
        cls, ast: Union[str, list], env: dict[str, "Program"]
    ) -> "Program":
        if isinstance(ast, str):
            if ast.startswith("$"):
                return Variable(int(ast[1:]))
            try:
                return value(int(ast)) if "." not in ast else value(float(ast))
            except ValueError:
                return value(ast)

        if not ast:
            raise SyntaxError("Empty expression")

        if ast[0] == "lam":
            return cls._build_from_ast(ast[1], env)

        func_name = ast[0]
        args = ast[1:]

        if func_name not in env:
            raise NameError(f"Unknown function '{func_name}' in environment")

        func_prog = env[func_name]
        parsed_args = [cls._build_from_ast(arg, env) for arg in args]
        return func_prog(*parsed_args)

    @property
    def func_name(self) -> str:
        return self._func.__name__

    def eval(self) -> R:
        """
        Evaluate this program to produce a result.

        Raises:
            VariableEvalError: If the program contains unsubstituted variables
        """
        if any(isinstance(arg, Variable) for arg in self._args):
            raise VariableEvalError("Cannot evaluate program with free variables.")

        evaluated_args = [arg.eval() for arg in self._args]
        return self._func(*evaluated_args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Program):
            return False
        if self._func != other._func:
            return False
        if len(self._args) != len(other._args):
            return False
        return all(a == b for a, b in zip(self._args, other._args))

    def __hash__(self) -> int:
        return hash((self._func, tuple(hash(arg) for arg in self._args)))

    def _renumber_variables(self) -> int:
        variable_map = {}
        next_index = 0

        def assign_indices(node: Program) -> None:
            nonlocal next_index
            if isinstance(node, Variable):
                if node not in variable_map:
                    variable_map[node] = next_index
                    next_index += 1
                node.index = variable_map[node]
            else:
                for arg in node._args:
                    assign_indices(arg)

        assign_indices(self)
        return next_index

    def _shift_variable_indices(self, shift: int) -> None:
        for arg in self._args:
            if isinstance(arg, Variable):
                arg.index += shift
            else:
                arg._shift_variable_indices(shift)

    def _collect_variables(self) -> set[int]:
        variables = set()
        for arg in self._args:
            if isinstance(arg, Variable):
                variables.add(arg.index)
            else:
                variables.update(arg._collect_variables())
        return variables

    def _validate_primitive_count(
        self, primitives: tuple, variables: list[int]
    ) -> None:
        if len(primitives) > len(variables):
            raise ValueError(
                f"Too many primitives: expected at most {len(variables)}, "
                f"got {len(primitives)}"
            )

    def _create_substitution_map(
        self, primitives: tuple, variables: list[int]
    ) -> dict[int, "Program"]:
        substitution_map = {}

        for i, var_index in enumerate(variables):
            if i < len(primitives):
                value = primitives[i]
                if value is Ellipsis:
                    substitution_map[var_index] = Variable(var_index)
                elif isinstance(value, Program):
                    substitution_map[var_index] = value
                else:
                    raise InvalidArgumentTypeError()
            else:
                # Keep unsubstituted variables
                substitution_map[var_index] = Variable(var_index)

        return substitution_map

    def _substitute_variables(
        self, substitution_map: dict[int, "Program"]
    ) -> "Program[R]":
        def substitute_node(node: Program) -> Program:
            if isinstance(node, Variable):
                return deepcopy(substitution_map.get(node.index, node))

            new_node = Program(node._func)
            new_node._args = [substitute_node(arg) for arg in node._args]
            new_node.__name__ = node.__name__
            return new_node

        return substitute_node(self)

    @property
    def cost(self) -> int:
        return self._cost


class Variable(Program):
    """
    Represents a variable (hole) in a program that can be substituted.

    Variables use De Bruijn indexing for consistent substitution.
    """

    def __init__(self, index: int, cost: int = 100):
        super().__init__(lambda: None, cost)
        self.index = index
        self.__name__ = f"<Variable: ${index}>"

    @property
    def name(self) -> str:
        return self.__name__

    def eval(self):
        """Variables cannot be evaluated directly."""
        raise VariableEvalError(self.__name__)

    def __str__(self) -> str:
        return f"${self.index}"

    def __repr__(
        self, indent: str = "", is_last: bool = True, is_root: bool = True
    ) -> str:
        """Generate string representation for tree display."""
        if is_root:
            return f"<Variable: ${self.index}>"

        connector = "└── " if is_last else "├── "
        return f"{indent}{connector}<Variable: ${self.index}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and self.index == other.index

    def __hash__(self) -> int:
        return hash(("Variable", self.index))


def value(value: R, cost: int = 100) -> Program[R]:
    """
    Create a program that represents a constant value.

    Args:
        value: The constant value to wrap

    Returns:
        Program that evaluates to the given value
    """
    program = Program[R](lambda: value, cost)
    program.__name__ = f"<value: {value}>"
    return program


def program(func: Callable[P, R], cost: int = 1) -> Program[R]:
    """
    Decorator to convert a function into a Program.

    Args:
        func: Function to convert

    Returns:
        Program wrapping the function
    """
    return Program(func, cost)
