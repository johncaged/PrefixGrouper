"""
Typing compatibility with different Python versions.
"""

import sys
from typing import *

if sys.version_info < (3, 8):
    try:
        from typing_extensions import (
            Literal,
            Protocol,
            SupportsIndex,
        )
    except Exception:
        print(
            "Loading ``typing_extensions`` module failed. "
            "Please make sure you have installed it correctly."
        )
        raise

if sys.version_info >= (3, 9):
    from builtins import (
        dict as Dict,
        list as List,
        set as Set,
        frozenset as FrozenSet,
        tuple as Tuple,
        type as Type,
        # for compatibility for Python 2.x
        str as Text,
    )

    from collections.abc import Sequence as Sequence
