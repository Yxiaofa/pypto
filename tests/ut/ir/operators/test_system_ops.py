# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for system operations, including sync_all."""

import pypto.language as pl
from pypto import ir
from pypto.ir.op import system

class TestSystemOps:
    """Test suite for system operations."""

    def test_sync_all_basic(self):
        """Test sync_all operation with default parameters (hardware sync)."""

        @pl.function
        def test_func(a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.system.sync_all()
            return a

        ir_str = str(test_func)
        assert "system.sync_all" in ir_str

    def test_sync_all_aiv_only_true(self):
        """Test sync_all with aiv_only=True."""

        @pl.function
        def test_func(a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.system.sync_all(aiv_only=True)
            return a

        ir_str = str(test_func)
        assert "system.sync_all" in ir_str
        assert "aiv_only" in ir_str

    def test_sync_all_aiv_only_false(self):
        """Test sync_all with aiv_only=False (mix mode)."""

        @pl.function
        def test_func(a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.system.sync_all(aiv_only=False)
            return a

        ir_str = str(test_func)
        assert "system.sync_all" in ir_str

    def test_sync_all_with_pipes(self):
        """Test sync_all with trigger_pipe and wait_pipe parameters."""

        @pl.function
        def test_func(a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.system.sync_all(
                aiv_only=True,
                trigger_pipe=pl.PipeType.MTE3,
                wait_pipe=pl.PipeType.ALL
            )
            return a

        ir_str = str(test_func)
        assert "system.sync_all" in ir_str

    def test_sync_all_with_all_pipes(self):
        """Test sync_all with PIPE_ALL for both pipes."""

        @pl.function
        def test_func(a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.system.sync_all(
                aiv_only=True,
                trigger_pipe=pl.PipeType.ALL,
                wait_pipe=pl.PipeType.ALL
            )
            return a

        ir_str = str(test_func)
        assert "system.sync_all" in ir_str

    def test_sync_all_multiple_calls(self):
        """Test multiple sync_all calls in same function."""

        @pl.function
        def test_func(a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            pl.system.sync_all()
            pl.system.sync_all(aiv_only=True)
            pl.system.sync_all(aiv_only=False)
            return a

        ir_str = str(test_func)
        assert ir_str.count("system.sync_all") == 3

    def test_sync_all_with_computation(self):
        """Test sync_all combined with computation."""

        @pl.function
        def test_func(
            a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            c: pl.Tensor[[64, 64], pl.FP32] = pl.add(a, b)
            pl.system.sync_all()
            d: pl.Tensor[[64, 64], pl.FP32] = pl.mul(c, c)
            return d

        ir_str = str(test_func)
        assert "system.sync_all" in ir_str
        assert "tensor.add" in ir_str
        assert "tensor.mul" in ir_str


class TestSystemOpsIR:
    """Test suite for system operations at IR level."""

    def test_sync_all_ir_creation(self):
        """Test creating sync_all operation directly at IR level."""
        call = system.sync_all(aiv_only=True)

        assert isinstance(call, ir.Call)
        assert call.op.name == "system.sync_all"

    def test_sync_all_ir_with_workspaces(self):
        """Test sync_all IR operation - hardware sync only (no workspaces)."""
        call = system.sync_all(aiv_only=True)

        assert isinstance(call, ir.Call)
        assert call.op.name == "system.sync_all"

    def test_sync_all_ir_with_pipes(self):
        """Test sync_all IR operation with pipe parameters."""
        call = system.sync_all(
            aiv_only=True,
            trigger_pipe=ir.PipeType.MTE3,
            wait_pipe=ir.PipeType.ALL,
        )

        assert isinstance(call, ir.Call)
        assert call.op.name == "system.sync_all"

    def test_sync_all_ir_default_params(self):
        """Test sync_all IR operation with default parameters."""
        call = system.sync_all()

        assert isinstance(call, ir.Call)
        assert call.op.name == "system.sync_all"

    def test_other_system_ops(self):
        """Test other system operations still work."""
        bar_v_call = system.bar_v()
        bar_m_call = system.bar_m()
        bar_all_call = system.bar_all()

        assert bar_v_call.op.name == "system.bar_v"
        assert bar_m_call.op.name == "system.bar_m"
        assert bar_all_call.op.name == "system.bar_all"
