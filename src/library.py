from .program import Program

from random import sample
from typing import Iterable, overload, Literal

from stitch_core import compress


class Library:
    programs: set[Program] = set()

    def __init__(self, *primitives: Program):
        self.register(*primitives)

    def register(self, *programs: Program) -> "Library":
        self.programs.update(programs)
        return self

    def remove(self, *programs: Program) -> "Library":
        self.programs.difference_update(programs)
        return self

    @overload
    def sample(
        self, k: Literal[1] = 1, counts: Iterable[int] | None = None
    ) -> Program: ...

    @overload
    def sample(self, k: int, counts: Iterable[int] | None = None) -> list[Program]: ...

    def sample(
        self, k: int = 1, counts: Iterable[int] | None = None
    ) -> Program | list[Program]:
        result = sample(self.programs, k, counts=counts)
        if k == 1:
            return result[0]
        return result

    def compress(self, *programs: Program) -> "Library":
        compression_result = compress(
            [str(p) for p in list(programs)], iterations=1, max_arity=2
        )
        abstractions = [
            Program.parse(abstraction.body.replace("#", "$"), self.to_dict())
            for abstraction in compression_result.abstractions
        ]
        print(abstractions)
        self.programs.update(abstractions)
        return self

    def to_dict(self) -> dict[str, Program]:
        return {p.func_name: p for p in self.programs}
