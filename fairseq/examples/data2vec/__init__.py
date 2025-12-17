# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("data2vec." + module)

for file in os.listdir(os.path.join(os.path.dirname(__file__), "tasks")):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("data2vec.tasks." + module)

for file in os.path.join(os.path.dirname(__file__), "models"):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("data2vec.models." + module)

for file in os.path.join(os.path.dirname(__file__), "criterions"):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("data2vec.criterions." + module)

