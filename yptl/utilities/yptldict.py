from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import Type

from pytorch_lightning.utilities.parsing import AttributeDict


class YPTLDict(AttributeDict):  # noqa: D101
    def __init__(self, init_dict: None | dict = None) -> None:  # noqa: D107
        super().__init__()
        if init_dict:
            for k, v in init_dict.items():
                self[k] = v

        if "type" not in self:
            msg = f"YPTLDict is missing required keyword 'type'.\n\n”YPTLDict: {self}"
            raise KeyError(msg)

        if "args" not in self:
            self.args = {}

        if not self.args:
            self.args = {}

        if not isinstance(self.args, dict):
            msg = f"In a YPTLDict 'args' must be a dictionary.\n\nYPTLDict: {self}"
            raise TypeError(msg)

    @classmethod
    def from_type_with_args(cls: Type[YPTLDict], type_name: str, args: None | dict = None) -> YPTLDict:  # noqa: D102
        if not args:
            args = {}
        return cls({"type": type_name, "args": args})
