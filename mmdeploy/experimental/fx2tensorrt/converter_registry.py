import logging
from typing import Any, Callable, Dict


def eval_with_import(path: str) -> Any:
    """Evaluate the string as Python script.

    Args:
        path (str): The path to evaluate.

    Returns:
        Any: The result of evaluate.
    """
    split_path = path.split('.')
    for i in range(len(split_path), 0, -1):
        try:
            exec('import {}'.format('.'.join(split_path[:i])))
            break
        except Exception:
            continue
    return eval(path)


class ConverterRegistry:
    """Registry for converter."""

    def __init__(self, default_converter=None) -> None:
        self._converter_map: Dict[Callable, Any] = {}
        self.default_converter = default_converter

    def set_default_converter(self, converter: Callable):
        self.default_converter = converter

    def register_converter(self, func_name: str, **kwargs):
        """Decorator used to register converter between fx and engine.

        Args:
            func_name (str): function name to be registered.

        Returns:
            wrap function of the converter.
        """
        func = eval_with_import(func_name)

        def wrap(converter: Callable):
            self._converter_map[func] = {'converter': converter, **kwargs}
            return converter

        return wrap

    def get_converter(self, func: Callable):
        """Get the converter which has been registered.

        Args:
            func (Callable): The fx Callable object, could be function or
            method of supported classes.

        Returns:
            converter (Callable): The converter of `func`.
        """
        if func not in self._converter_map:
            if self.default_converter is None:
                raise NotImplementedError(f'No converter for {func}.')
            else:
                logging.warning(
                    f'No converter for {func}, use default converter.')
                return self.default_converter
        return self._converter_map[func]['converter']


TRT_REGISTRY = ConverterRegistry()
