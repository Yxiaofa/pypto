# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual (non-SSA) IR operations for PyPTO.

IR-level functions for manual ops with special-case argument handling.
Called from the parser via:
    if hasattr(ir_op.manual, op_name):
        return getattr(ir_op.manual, op_name)(*args, **kwargs, span=span)
"""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, Span, ConstInt

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def load(
    out: Expr,
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
    layout: str | None = None,
) -> Call:
    """Build manual.load IR call.

    Args:
        out: Pre-allocated destination tile.
        tensor: Source tensor expression.
        offsets: Offsets tuple or sequence.
        span: Optional source span.
        layout: Tensor memory layout, "ND" (row-major) or "DN" (column-major).

    Returns:
        Call expression for manual.load.
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    kwargs: dict = {}
    if layout is not None:
        kwargs["layout"] = layout
    return _ir_core.create_op_call(
        "manual.load", [tensor, offsets_tuple, out], kwargs, actual_span
    )


def store(
    out: Expr,
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Build manual.store IR call.

    Args:
        out: Out tensor.
        tile: Source tile expression.
        offsets: Offsets tuple or sequence.
        span: Optional source span.

    Returns:
        Call expression for manual.store.
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    return _ir_core.create_op_call(
        "manual.store", [tile, offsets_tuple, out], {}, actual_span,
    )


def load_tile(
    out: Expr,
    tensor: Expr,
    tile_offsets: _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Build manual.load with abs_offsets = tile_offsets * shapes.

    Args:
        out: Pre-allocated destination tile.
        tensor: Source tensor.
        tile_offsets: MakeTuple of tile-relative offsets.
        span: Optional source span.

    Returns:
        Call expression for manual.load with absolute offsets.
    """
    actual_span = _get_span_or_capture(span)
    offsets = []
    shapes = out.type.shape
    for tile_offset, shape in zip(tile_offsets.elements, shapes):
        if isinstance(tile_offset, ConstInt) and isinstance(shape, ConstInt):
            offset = ConstInt(tile_offset.value * shape.value, DataType.INT64, actual_span)
        else:
            offset = _ir_core.Mul(tile_offset, shape, DataType.INT64, actual_span)
        offsets.append(offset)
    offsets_tuple = _ir_core.MakeTuple(offsets, actual_span)
    kwargs = {}
    return _ir_core.create_op_call(
        "manual.load", [tensor, offsets_tuple, out], kwargs, actual_span
    )


def store_tile(
    out: Expr,
    tile: Expr,
    tile_offsets: _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Build manual.store with abs_offsets = tile_offsets * shapes.

    Args:
        output_tensor: Destination tensor.
        tile: Source tile.
        tile_offsets: MakeTuple of tile-relative offsets.
        span: Optional source span.

    Returns:
        Call expression for manual.store with absolute offsets.
    """
    actual_span = _get_span_or_capture(span)
    offsets = []
    shapes = tile.type.shape
    for tile_offset, shape in zip(tile_offsets.elements, shapes):
        if isinstance(tile_offset, ConstInt) and isinstance(shape, ConstInt):
            offset = ConstInt(tile_offset.value * shape.value, DataType.INT64, actual_span)
        else:
            offset = _ir_core.Mul(tile_offset, shape, DataType.INT64, actual_span)
        offsets.append(offset)
    offsets_tuple = _ir_core.MakeTuple(offsets, actual_span)

    return _ir_core.create_op_call(
        "manual.store", [tile, offsets_tuple, out], {}, actual_span
    )
