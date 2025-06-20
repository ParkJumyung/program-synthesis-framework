class ProgramSynthesisError(Exception):
    """Base class for all Program-related errors."""


class NotCallableError(ProgramSynthesisError):
    def __init__(self, value):
        super().__init__(f"Expected a callable, but got: {type(value).__name__}")


class TooFewArgumentsError(ProgramSynthesisError):
    def __init__(self, program_name: str, expected: int, received: int):
        super().__init__(
            f"{program_name} expected at least {expected} arguments, but got {received}."
        )


class InvalidArgumentTypeError(ProgramSynthesisError):
    def __init__(self):
        super().__init__("All arguments must be instances of Program.")


class ProgramTypeError(ProgramSynthesisError):
    def __init__(self, program_name: str, arg_index: int, expected, actual):
        super().__init__(
            f"Argument {arg_index} to {program_name} expects return type {expected}, "
            f"but got {actual}"
        )


class VariableEvalError(ProgramSynthesisError):
    def __init__(self, variable_name: str):
        super().__init__(f"Variable {variable_name} cannot be evaluated")
