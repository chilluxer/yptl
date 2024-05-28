from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import Type

from pytorch_lightning.utilities.parsing import AttributeDict


class YPTLDict(AttributeDict):  # noqa: D101
    def __init__(self, input_dict: None | dict = None) -> None:  # noqa: D107
        if "type" not in input_dict:
            msg = f"YPTLDict is missing required keyword 'type'.\n\n”YPTLDict: {input_dict}"
            raise KeyError(msg)

        if "args" not in input_dict:
            input_dict["args"] = {}

        if not input_dict["args"]:
            input_dict["args"] = {}

        if not isinstance(input_dict["args"], dict):
            msg = (
                f"In a YPTLDict 'args' must be a dictionary.\n\nYPTLDict: {input_dict}"
            )
            raise TypeError(msg)

        # This line must be at the end of the init method that all modifications to the input_dict are copied into self
        super().__init__(input_dict)

    @classmethod
    def from_type_with_args(
        cls: Type[YPTLDict], type_name: str, args: None | dict = None
    ) -> YPTLDict:  # noqa: D102
        if not args:
            args = {}
        return cls({"type": type_name, "args": args})
