from .io_example import IOExample, I, O
from .program import Program, value

from typing import Sequence, Optional, Iterator, Generic


class Task(Generic[I, O]):
    def __init__(self, io_examples: Sequence[IOExample[I, O]]):
        self.io_examples: list[IOExample[I, O]] = list(io_examples)
        self.solution_program: Optional[Program] = None

    def validate_program(self, program: Program) -> bool:
        is_valid = all(
            program.apply(value(input)).eval() == output
            for input, output in self.io_examples
        )
        if is_valid:
            self.solution_program = program
        return is_valid

    def __iter__(self) -> Iterator[IOExample[I, O]]:
        return iter(self.io_examples)

    def inputs(self) -> Iterator[I]:
        return (input for input, _ in self.io_examples)

    def outputs(self) -> Iterator[O]:
        return (output for _, output in self.io_examples)
