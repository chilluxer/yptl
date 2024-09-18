from dataclasses import dataclass


@dataclass(frozen=True)
class MyClass:
    a: int
    b: str
    d: dict
