from typing import Generic, TypeVar, NamedTuple

I = TypeVar("I")
O = TypeVar("O")


class IOExample(NamedTuple, Generic[I, O]):
    input: I
    output: O
