from . import utils  # noqa: F401  must be imported before dependency/methods
from .benchmark import compare
from .core import CompressionError, compress
from .finetune import finetune
from .registry import describe_methods, list_methods, register
from . import methods  # noqa: F401,E402

__version__ = "0.1.0"

__all__ = ["compress", "compare", "finetune", "list_methods", "describe_methods", "register", "CompressionError"]
