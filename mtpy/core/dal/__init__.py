from importlib import import_module
import sys


def __getattr__(name):
    package = sys.modules[__name__]
    module = import_module('.' + name, __name__)
    attribute = getattr(module, name)

    setattr(package, name, attribute)

    return getattr(package, name)
