"""Modified from vedaseg"""

import inspect


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        items = list(self._module_dict.keys())
        format_str = (f"{self.__class__.__name__}"
                      f"(name={self._name}, items={items})")
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """
            Register a module.
        Args:
            module_class (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, "
                            f"but got {type(module_class)}")
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f"{module_name} already registered in {self.name}")
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls
