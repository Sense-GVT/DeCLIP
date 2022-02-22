"""The lightly.models.modules package provides reusable modules.

This package contains reusable modules such as the NNmemoryBankModule which
can be combined with any lightly model.

"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from .nn_memory_bank import NNMemoryBankModule
from .memory_bank_cuda import MemoryBankModule
