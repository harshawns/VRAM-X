# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Env Environment."""

from .generator import generate_episode_state
from .models import MyAction, MyObservation

__all__ = [
    "MyAction",
    "MyObservation",
    "MyEnv",
    "generate_episode_state",
]


def __getattr__(name: str):
    """Defer client import so non-client utilities can be imported independently."""
    if name == "MyEnv":
        from .client import MyEnv

        return MyEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
