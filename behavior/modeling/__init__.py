from __future__ import annotations

import numpy as np
from behavior import OperatingUnit

# OUs that we don't train models for.
BLOCKED_OUS = [
    OperatingUnit.AfterQueryTrigger
]
