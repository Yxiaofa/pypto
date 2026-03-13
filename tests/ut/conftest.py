# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared fixtures for unit tests."""
import os
import pytest
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def set_npu_arch():
    """Set npu_arch environment variable for all tests.
    
    This ensures the backend can access the NPU architecture information
    during test execution.
    """
    original_value = os.environ.get("npu_arch")
    os.environ["npu_arch"] = "dav-c220"
    yield
    if original_value is not None:
        os.environ["npu_arch"] = original_value
    else:
        os.environ.pop("npu_arch", None)

def pass_verification_context():
    """Enable BEFORE_AND_AFTER property verification for all pass executions.

    This ensures that for every pass run in a test:
    - Before execution, its required properties are verified.
    - After execution, its produced properties are verified.

    This helps keep pass property declarations accurate.
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield
