/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../module.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/common.h"
#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/serialization/deserializer.h"
#include "pypto/ir/serialization/serializer.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/op_conversion_registry.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
#include "pypto/ir/type.h"

namespace py = pybind11;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)
using pypto::DataType;

template <typename T>
bool TryConvertAnyToPy(const std::any& value, py::object& out) {
  if (value.type() != typeid(T)) {
    return false;
  }
  out = py::cast(AnyCastRef<T>(value, "converting to Python"));
  return true;
}

template <typename... Ts>
py::object AnyToPyObject(const std::any& value, const std::string& key) {
  py::object out;
  if ((TryConvertAnyToPy<Ts>(value, out) || ...)) {
    return out;
  }
  throw pypto::TypeError("Attribute '" + key + "' has unsupported type");
}

// Helper to bind a single field using reflection
template <typename ClassType, typename PyClassType, typename FieldDesc>
void BindField(PyClassType& py_class, const FieldDesc& desc) {
  py_class.def_readonly(desc.name, desc.field_ptr);
}

// Helper to bind all fields from a tuple of field descriptors
template <typename ClassType, typename PyClassType, typename DescTuple, std::size_t... Is>
void BindFieldsImpl(PyClassType& py_class, const DescTuple& descriptors, std::index_sequence<Is...>) {
  (BindField<ClassType>(py_class, std::get<Is>(descriptors)), ...);
}

// Main function to bind all fields using reflection
template <typename ClassType, typename PyClassType>
void BindFields(PyClassType& py_class) {
  constexpr auto descriptors = ClassType::GetFieldDescriptors();
  constexpr auto num_fields = std::tuple_size_v<decltype(descriptors)>;
  BindFieldsImpl<ClassType>(py_class, descriptors, std::make_index_sequence<num_fields>{});
}

// Helper function to convert py::dict to vector<pair<string, any>>
std::vector<std::pair<std::string, std::any>> ConvertKwargsDict(const py::dict& kwargs_dict) {
  std::vector<std::pair<std::string, std::any>> kwargs;
  for (auto item : kwargs_dict) {
    std::string key = py::cast<std::string>(item.first);

    // Try to cast to common types.
    // NOTE: Check enum-like values BEFORE plain int, and bool BEFORE int.
    if (py::isinstance<DataType>(item.second)) {
      kwargs.emplace_back(key, py::cast<DataType>(item.second));
    } else if (py::isinstance<MemorySpace>(item.second)) {
      kwargs.emplace_back(key, py::cast<MemorySpace>(item.second));
    } else if (py::isinstance<TilePad>(item.second)) {
      kwargs.emplace_back(key, static_cast<int>(py::cast<TilePad>(item.second)));
    } else if (py::isinstance<PipeType>(item.second)) {
      // Cast enum to int for storage
      kwargs.emplace_back(key, static_cast<int>(py::cast<PipeType>(item.second)));
    } else if (py::isinstance<CoreType>(item.second)) {
      // Cast enum to int for storage
      kwargs.emplace_back(key, static_cast<int>(py::cast<CoreType>(item.second)));
    } else if (py::isinstance<py::bool_>(item.second)) {
      kwargs.emplace_back(key, py::cast<bool>(item.second));
    } else if (py::isinstance<py::int_>(item.second)) {
      kwargs.emplace_back(key, py::cast<int>(item.second));
    } else if (py::isinstance<py::str>(item.second)) {
      kwargs.emplace_back(key, py::cast<std::string>(item.second));
    } else if (py::isinstance<py::float_>(item.second)) {
      kwargs.emplace_back(key, py::cast<double>(item.second));
    } else {
      throw pypto::TypeError("Unsupported kwarg type for key: " + key);
    }
  }
  return kwargs;
}

void BindIR(py::module_& m) {
  py::module_ ir = m.def_submodule("ir", "PyPTO IR (Intermediate Representation) module");

  // Span - value type, copy semantics
  py::class_<Span>(ir, "Span", "Source location information tracking file, line, and column positions")
      .def(py::init<std::string, int, int, int, int>(), py::arg("filename"), py::arg("begin_line"),
           py::arg("begin_column"), py::arg("end_line") = -1, py::arg("end_column") = -1,
           "Create a source span")
      .def("to_string", &Span::to_string, "Convert span to string representation")
      .def("is_valid", &Span::is_valid, "Check if the span has valid coordinates")
      .def_static("unknown", &Span::unknown,
                  "Create an unknown/invalid span for cases where source location is unavailable")
      .def("__repr__", &Span::to_string)
      .def("__str__", &Span::to_string)
      .def_readonly("filename", &Span::filename_, "Source filename")
      .def_readonly("begin_line", &Span::begin_line_, "Beginning line (1-indexed)")
      .def_readonly("begin_column", &Span::begin_column_, "Beginning column (1-indexed)")
      .def_readonly("end_line", &Span::end_line_, "Ending line (1-indexed)")
      .def_readonly("end_column", &Span::end_column_, "Ending column (1-indexed)");

  // Op - operation/function
  py::class_<Op, std::shared_ptr<Op>>(
      ir, "Op",
      "Represents callable operations in the IR. Stores the schema of allowed kwargs (key -> type "
      "mapping). Actual kwarg values are stored per-Call instance in Call.kwargs")
      .def(py::init<std::string>(), py::arg("name"), "Create an operation with the given name")
      .def_readonly("name", &Op::name_, "Operation name")
      .def("has_attr", &Op::HasAttr, py::arg("key"), "Check if a kwarg is registered in the schema")
      .def("get_attr_keys", &Op::GetAttrKeys, "Get all registered kwarg keys from the schema")
      .def_property_readonly(
          "pipe", [](const Op& self) -> std::optional<PipeType> { return self.GetPipe(); },
          "Pipeline type (optional)");

  // GlobalVar - global function reference
  py::class_<GlobalVar, Op, std::shared_ptr<GlobalVar>>(
      ir, "GlobalVar",
      "Global variable reference for functions in a program. "
      "Can be used in Call expressions to invoke functions within the same program.")
      .def(py::init<std::string>(), py::arg("name"),
           "Create a global variable reference with the given name");

  // Type - abstract base, const shared_ptr
  auto type_class = py::class_<Type, std::shared_ptr<Type>>(ir, "Type", "Base class for type representations");
  BindFields<Type>(type_class);
  type_class.def(
      "__str__", [](const TypePtr& self) { return PythonPrint(self, "pl"); },
      "Python-style string representation");
  type_class.def(
      "__eq__", [](const TypePtr& self, const TypePtr& other) { return structural_equal(self, other); },
      "Equality comparison");

  // UnknownType - const shared_ptr
  auto unknown_type_class =
      py::class_<UnknownType, Type, std::shared_ptr<UnknownType>>(ir, "UnknownType",
                                                                        "Unknown or unspecified type representation");
  unknown_type_class.def(py::init<>(), "Create an unknown type");
  unknown_type_class.def_static(
      "get", []() { return GetUnknownType(); }, "Get the singleton UnknownType instance");
  BindFields<UnknownType>(unknown_type_class);

  // ScalarType - const shared_ptr
  auto scalar_type_class =
      py::class_<ScalarType, Type, std::shared_ptr<ScalarType>>(ir, "ScalarType",
                                                                      "Scalar type representation");
  scalar_type_class.def(py::init<DataType>(), py::arg("dtype"), "Create a scalar type");
  BindFields<ScalarType>(scalar_type_class);

  // PtrType - const shared_ptr
  auto ptr_type_class =
      py::class_<PtrType, Type, std::shared_ptr<PtrType>>(ir, "PtrType",
                                                                 "Pointer type representation (!pto.ptr<dtype>)");
  ptr_type_class.def(py::init<DataType>(), py::arg("dtype"), "Create a pointer type");
  BindFields<PtrType>(ptr_type_class);

  // IRNode - abstract base, const shared_ptr
  auto irnode_class =
      py::class_<IRNode, std::shared_ptr<IRNode>>(ir, "IRNode", "Base class for all IR nodes");
  BindFields<IRNode>(irnode_class);
  irnode_class
      .def(
          "same_as", [](const IRNodePtr& self, const IRNodePtr& other) { return self == other; },
          py::arg("other"), "Check if this IR node is the same as another IR node.")
      .def(
          "__str__",
          [](const IRNodePtr& self) {
            // Use unified PythonPrint API with default "pl" prefix
            return PythonPrint(self, "pl");
          },
          "Python-style string representation")
      .def(
          "as_python",
          [](const IRNodePtr& self, const std::string& prefix) { return PythonPrint(self, prefix); },
          py::arg("prefix") = "pl",
          "Convert to Python-style string representation.\n\n"
          "Args:\n"
          "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')");

  // Expr - abstract base, const shared_ptr
  auto expr_class =
      py::class_<Expr, IRNode, std::shared_ptr<Expr>>(ir, "Expr", "Base class for all expressions");
  BindFields<Expr>(expr_class);

  // ShapedType - abstract base for types with shape and optional memref
  auto shaped_type_class =
      py::class_<ShapedType, Type, std::shared_ptr<ShapedType>>(ir, "ShapedType",
                                                                      "Base class for shaped types (tensors and tiles)");
  BindFields<ShapedType>(shaped_type_class);
  shaped_type_class.def(
      "shares_memref_with",
      [](const ShapedTypePtr& self, const ShapedTypePtr& other) {
        if (!self->memref_.has_value() || !other->memref_.has_value()) {
          return false;
        }
        return self->memref_.value().get() == other->memref_.value().get();
      },
      py::arg("other"), "Check if this ShapedType shares the same MemRef object with another ShapedType");

  // TensorLayout enum - must be before TensorView and TensorType
  py::enum_<TensorLayout>(ir, "TensorLayout", "Tensor layout enumeration")
      .value("ND", TensorLayout::ND, "ND layout")
      .value("DN", TensorLayout::DN, "DN layout")
      .value("NZ", TensorLayout::NZ, "NZ layout")
      .export_values();

  // TensorView - struct for tensor view information - must be before TensorType
  py::class_<TensorView>(ir, "TensorView", "Tensor view representation with stride and layout")
      .def(py::init<>(), "Create an empty tensor view")
      .def(py::init<const std::vector<ExprPtr>&, TensorLayout>(), py::arg("stride"), py::arg("layout"),
           "Create a tensor view with stride and layout")
      .def_readwrite("stride", &TensorView::stride, "Stride for each dimension")
      .def_readwrite("layout", &TensorView::layout, "Tensor layout type")
      .def_readwrite("ptr", &TensorView::ptr,
                     "Source pointer ExprPtr (set for ptr.make_tensor-created views; None otherwise).");

  // TensorType - const shared_ptr
  auto tensor_type_class =
      py::class_<TensorType, ShapedType, std::shared_ptr<TensorType>>(ir, "TensorType",
                                                                             "Tensor type representation");
  tensor_type_class.def(py::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>>(),
                        py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
                        "Create a tensor type");
  tensor_type_class.def(py::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>>(),
                        py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
                        "Create a tensor type");
  tensor_type_class.def(
      py::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
      py::arg("tensor_view") = py::none(),
      "Create a tensor type with optional memory reference and tensor view");
  tensor_type_class.def(
      py::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>, std::optional<TensorView>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
      py::arg("tensor_view") = py::none(),
      "Create a tensor type with constant shape, optional memory reference and tensor view");
  BindFields<TensorType>(tensor_type_class);

  // TileType - const shared_ptr
  auto tile_type_class =
      py::class_<TileType, ShapedType, std::shared_ptr<TileType>>(
          ir, "TileType", "Tile type representation (multi-dimensional tensor)");
  tile_type_class.def(
      py::init<const std::vector<ExprPtr>&, DataType, std::optional<MemRefPtr>, std::optional<TileView>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
      py::arg("tile_view") = py::none(),
      "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
  tile_type_class.def(
      py::init<const std::vector<int64_t>&, DataType, std::optional<MemRefPtr>, std::optional<TileView>>(),
      py::arg("shape"), py::arg("dtype"), py::arg("memref") = py::none(),
      py::arg("tile_view") = py::none(),
      "Create a tile type (supports multi-dimensional tensors; code generation has constraints)");
  BindFields<TileType>(tile_type_class);

  // TupleType - const shared_ptr
  auto tuple_type_class =
      py::class_<TupleType, Type, std::shared_ptr<TupleType>>(
          ir, "TupleType", "Tuple type representation (contains multiple types)");
  tuple_type_class.def(py::init<const std::vector<TypePtr>&>(), py::arg("types"),
                       "Create a tuple type from a list of types");
  BindFields<TupleType>(tuple_type_class);

  // MemorySpace enum
  py::enum_<MemorySpace>(ir, "MemorySpace", "Memory space enumeration")
      .value("DDR", MemorySpace::DDR, "DDR memory (off-chip)")
      .value("Vec", MemorySpace::Vec, "Vector/unified buffer (on-chip)")
      .value("Mat", MemorySpace::Mat, "Matrix/L1 buffer")
      .value("Left", MemorySpace::Left, "Left matrix operand buffer")
      .value("Right", MemorySpace::Right, "Right matrix operand buffer")
      .value("Acc", MemorySpace::Acc, "Accumulator buffer")
      .export_values();

  // PipeType enum
  py::enum_<PipeType>(ir, "PipeType", py::arithmetic(), "Pipeline type enumeration")
      .value("MTE1", PipeType::MTE1, "Memory Transfer Engine 1")
      .value("MTE2", PipeType::MTE2, "Memory Transfer Engine 2")
      .value("MTE3", PipeType::MTE3, "Memory Transfer Engine 3")
      .value("M", PipeType::M, "Matrix Unit")
      .value("V", PipeType::V, "Vector Unit")
      .value("S", PipeType::S, "Scalar Unit")
      .value("FIX", PipeType::FIX, "Fix Pipe")
      .value("ALL", PipeType::ALL, "All Pipes")
      .export_values();

  // CoreType enum
  py::enum_<CoreType>(ir, "CoreType", py::arithmetic(), "Core type enumeration")
      .value("VECTOR", CoreType::VECTOR, "Vector Core")
      .value("CUBE", CoreType::CUBE, "Cube Core")
      .export_values();

  // TileLayout enum - must be before TileView
  py::enum_<TileLayout>(ir, "TileLayout", "Tile layout enumeration")
      .value("none_box", TileLayout::none_box, "No layout constraint")
      .value("row_major", TileLayout::row_major, "Row-major layout")
      .value("col_major", TileLayout::col_major, "Column-major layout")
      .export_values();

  // TilePad enum - must be before TileView
  py::enum_<TilePad>(ir, "TilePad", "Tile pad mode enumeration")
      .value("null", TilePad::null, "No padding")
      .value("zero", TilePad::zero, "Zero padding")
      .value("max", TilePad::max, "Max value padding")
      .value("min", TilePad::min, "Min value padding")
      .export_values();

  // CompactMode enum - must be before TileView
  py::enum_<CompactMode>(ir, "CompactMode", "Compact mode for tile buffer")
      .value("null",         CompactMode::null,         "No compact mode")
      .value("normal",       CompactMode::normal,       "Normal compact mode")
      .value("row_plus_one", CompactMode::row_plus_one, "Row plus one compact mode")
      .export_values();

  // TileView - struct for tile view information
  py::class_<TileView>(
      ir, "TileView",
      "Tile view representation with valid shape, stride, start offset, layouts, fractal, and pad")
      .def(py::init<>(), "Create an empty tile view")
      .def(py::init<const std::vector<ExprPtr>&, const std::vector<ExprPtr>&, ExprPtr, TileLayout, TileLayout,
                    uint64_t, TilePad, CompactMode>(),
           py::arg("valid_shape"), py::arg("stride"), py::arg("start_offset"),
           py::arg("blayout") = TileLayout::row_major, py::arg("slayout") = TileLayout::none_box,
           py::arg("fractal") = static_cast<uint64_t>(512), py::arg("pad") = TilePad::null,
           py::arg("compact") = CompactMode::null,
           "Create a tile view with valid_shape, stride, start_offset, blayout, slayout, fractal, pad, and compact")
      .def_readwrite("valid_shape", &TileView::valid_shape, "Valid shape dimensions")
      .def_readwrite("stride", &TileView::stride, "Stride for each dimension")
      .def_readwrite("start_offset", &TileView::start_offset, "Starting offset")
      .def_readwrite("blayout", &TileView::blayout, "Block layout")
      .def_readwrite("slayout", &TileView::slayout, "Scatter layout")
      .def_readwrite("fractal", &TileView::fractal, "Fractal size")
      .def_readwrite("pad", &TileView::pad, "Pad mode")
      .def_readwrite("compact", &TileView::compact, "Compact mode");

  // Dynamic dimension constant
  ir.attr("DYNAMIC_DIM") = kDynamicDim;

  // OpRegistry
  ir.def(
      "create_op_call",
      [](const std::string& op_name, const std::vector<ExprPtr>& args, const Span& span) {
        return OpRegistry::GetInstance().Create(op_name, args, span);
      },
      py::arg("op_name"), py::arg("args"), py::arg("span"),
      "Create a Call expression (backward compatibility)");

  ir.def(
      "create_op_call",
      [](const std::string& op_name, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
         const Span& span) {
        // Convert Python dict to C++ vector<pair<string, any>> to preserve order
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
      },
      py::arg("op_name"), py::arg("args"), py::arg("kwargs"), py::arg("span"),
      "Create a Call expression with args and kwargs");

  ir.def(
      "is_op_registered",
      [](const std::string& op_name) { return OpRegistry::GetInstance().IsRegistered(op_name); },
      py::arg("op_name"), "Check if an operator is registered");

  ir.def(
      "get_op", [](const std::string& op_name) { return OpRegistry::GetInstance().GetOp(op_name); },
      py::arg("op_name"), "Get an operator instance by name");

  // Var - const shared_ptr
  auto var_class = py::class_<Var, Expr, std::shared_ptr<Var>>(ir, "Var", "Variable reference expression");

  var_class.def(
      py::init<const std::string&, const TypePtr&, const Span&>(), py::arg("name"), py::arg("type"),
      py::arg("span"),
      "Create a variable reference (memory reference is stored in ShapedType for Tensor/Tile types)");
  BindFields<Var>(var_class);

  // IterArg - const shared_ptr
  auto iterarg_class =
      py::class_<IterArg, Var, std::shared_ptr<IterArg>>(ir, "IterArg", "Iteration argument variable");
  iterarg_class.def(py::init<const std::string&, const TypePtr&, const ExprPtr&, const Span&>(),
                    py::arg("name"), py::arg("type"), py::arg("initValue"), py::arg("span"),
                    "Create an iteration argument with initial value");
  BindFields<IterArg>(iterarg_class);

  // MemRef - now inherits from Var (first-class expression)
  auto memref_class =
      py::class_<MemRef, Var, std::shared_ptr<MemRef>>(
          ir, "MemRef", "Memory reference variable for shaped types (inherits from Var)");
  memref_class
      .def(py::init<MemorySpace, ExprPtr, uint64_t, uint64_t, Span>(), py::arg("memory_space"),
           py::arg("addr"), py::arg("size"), py::arg("id"), py::arg("span") = Span::unknown(),
           "Create a memory reference with memory_space, addr, size, id, and span")
      .def_readwrite("memory_space_", &MemRef::memory_space_, "Memory space (DDR, Vec, Mat, Left, Right, Acc)")
      .def_readwrite("addr_", &MemRef::addr_, "Starting address expression")
      .def_readwrite("size_", &MemRef::size_, "Size in bytes (64-bit unsigned)")
      .def_readwrite("id_", &MemRef::id_, "Unique identifier for this MemRef instance");

  // ConstInt - const shared_ptr
  auto constint_class =
      py::class_<ConstInt, Expr, std::shared_ptr<ConstInt>>(ir, "ConstInt",
                                                                   "Constant integer expression");
  constint_class.def(py::init<int64_t, DataType, const Span&>(), py::arg("value"), py::arg("dtype"),
                     py::arg("span"), "Create a constant integer expression");
  BindFields<ConstInt>(constint_class);
  constint_class.def_property_readonly("dtype", &ConstInt::dtype, "Data type of the expression");

  // ConstFloat - const shared_ptr
  auto constfloat_class =
      py::class_<ConstFloat, Expr, std::shared_ptr<ConstFloat>>(ir, "ConstFloat",
                                                                       "Constant float expression");
  constfloat_class.def(py::init<double, DataType, const Span&>(), py::arg("value"), py::arg("dtype"),
                       py::arg("span"), "Create a constant float expression");
  BindFields<ConstFloat>(constfloat_class);
  constfloat_class.def_property_readonly("dtype", &ConstFloat::dtype, "Data type of the expression");

  // ConstBool - const shared_ptr
  auto constbool_class =
      py::class_<ConstBool, Expr, std::shared_ptr<ConstBool>>(ir, "ConstBool",
                                                                     "Constant boolean expression");
  constbool_class.def(py::init<bool, const Span&>(), py::arg("value"), py::arg("span"),
                      "Create a constant boolean expression");
  BindFields<ConstBool>(constbool_class);
  constbool_class.def_property_readonly("dtype", &ConstBool::dtype,
                                        "Data type of the expression (always BOOL)");

  // Call - const shared_ptr
  auto call_class =
      py::class_<Call, Expr, std::shared_ptr<Call>>(ir, "Call", "Function call expression");

  // Constructors without kwargs (backward compatibility)
  call_class.def(py::init<const OpPtr&, const std::vector<ExprPtr>&, const Span&>(), py::arg("op"),
                 py::arg("args"), py::arg("span"), "Create a function call expression");
  call_class.def(py::init<const OpPtr&, const std::vector<ExprPtr>&, const TypePtr&, const Span&>(),
                 py::arg("op"), py::arg("args"), py::arg("type"), py::arg("span"),
                 "Create a function call expression with explicit type");

  // Constructors with kwargs (using py::dict) - use factory functions
  call_class.def(
      py::init([](const OpPtr& op, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
                  const Span& span) -> std::shared_ptr<Call> {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        return std::make_shared<Call>(op, args, kwargs, span);
      }),
      py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("span"),
      "Create a function call expression with kwargs");

  call_class.def(
      py::init([](const OpPtr& op, const std::vector<ExprPtr>& args, const py::dict& kwargs_dict,
                  const TypePtr& type, const Span& span) -> std::shared_ptr<Call> {
        auto kwargs = ConvertKwargsDict(kwargs_dict);
        return std::make_shared<Call>(op, args, kwargs, type, span);
      }),
      py::arg("op"), py::arg("args"), py::arg("kwargs"), py::arg("type"), py::arg("span"),
      "Create a function call expression with kwargs and explicit type");

  BindFields<Call>(call_class);

  // Expose kwargs as a read-only property
  call_class.def_property_readonly(
      "kwargs",
      [](const CallPtr& self) {
        py::dict result;
        for (const auto& [key, value] : self->kwargs_) {
          if (value.type() == typeid(int)) {
            result[key.c_str()] = AnyCast<int>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(int64_t)) {
            result[key.c_str()] = AnyCast<int64_t>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(bool)) {
            result[key.c_str()] = AnyCast<bool>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(std::string)) {
            result[key.c_str()] = AnyCast<std::string>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(double)) {
            result[key.c_str()] = AnyCast<double>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(float)) {
            result[key.c_str()] = AnyCast<float>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(DataType)) {
            result[key.c_str()] = AnyCast<DataType>(value, "converting to Python: " + key);
          } else if (value.type() == typeid(TilePad)) {
            result[key.c_str()] = AnyCast<TilePad>(value, "converting to Python: " + key);
          }
        }
        return result;
      },
      "Keyword arguments (metadata) for this call");

  // MakeTuple - const shared_ptr
  auto make_tuple_class =
      py::class_<MakeTuple, Expr, std::shared_ptr<MakeTuple>>(ir, "MakeTuple",
                                                                     "Tuple construction expression");
  make_tuple_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("elements"),
                       py::arg("span"), "Create a tuple construction expression");
  BindFields<MakeTuple>(make_tuple_class);

  // TupleGetItemExpr - const shared_ptr
  auto tuple_get_item_class =
      py::class_<TupleGetItemExpr, Expr, std::shared_ptr<TupleGetItemExpr>>(
          ir, "TupleGetItemExpr", "Tuple element access expression");
  tuple_get_item_class.def(py::init<const ExprPtr&, int, const Span&>(), py::arg("tuple"), py::arg("index"),
                           py::arg("span"), "Create a tuple element access expression");
  BindFields<TupleGetItemExpr>(tuple_get_item_class);

  // BinaryExpr - abstract, const shared_ptr
  auto binaryexpr_class =
      py::class_<BinaryExpr, Expr, std::shared_ptr<BinaryExpr>>(ir, "BinaryExpr",
                                                                       "Base class for binary operations");
  BindFields<BinaryExpr>(binaryexpr_class);

  // UnaryExpr - abstract, const shared_ptr
  auto unaryexpr_class =
      py::class_<UnaryExpr, Expr, std::shared_ptr<UnaryExpr>>(ir, "UnaryExpr",
                                                                     "Base class for unary operations");
  BindFields<UnaryExpr>(unaryexpr_class);

// Macro to bind binary expression nodes
#define BIND_BINARY_EXPR(OpName, Description)                                                   \
  py::class_<OpName, BinaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)      \
      .def(py::init<const ExprPtr&, const ExprPtr&, DataType, const Span&>(), py::arg("left"), \
           py::arg("right"), py::arg("dtype"), py::arg("span"), "Create " Description);

  // Bind all binary expression nodes
  BIND_BINARY_EXPR(Add, "Addition expression (left + right)")
  BIND_BINARY_EXPR(Sub, "Subtraction expression (left - right)")
  BIND_BINARY_EXPR(Mul, "Multiplication expression (left * right)")
  BIND_BINARY_EXPR(FloorDiv, "Floor division expression (left // right)")
  BIND_BINARY_EXPR(FloorMod, "Floor modulo expression (left % right)")
  BIND_BINARY_EXPR(FloatDiv, "Float division expression (left / right)")
  BIND_BINARY_EXPR(Min, "Minimum expression (min(left, right))")
  BIND_BINARY_EXPR(Max, "Maximum expression (max(left, right))")
  BIND_BINARY_EXPR(Pow, "Power expression (left ** right)")
  BIND_BINARY_EXPR(Eq, "Equality expression (left == right)")
  BIND_BINARY_EXPR(Ne, "Inequality expression (left != right)")
  BIND_BINARY_EXPR(Lt, "Less than expression (left < right)")
  BIND_BINARY_EXPR(Le, "Less than or equal to expression (left <= right)")
  BIND_BINARY_EXPR(Gt, "Greater than expression (left > right)")
  BIND_BINARY_EXPR(Ge, "Greater than or equal to expression (left >= right)")
  BIND_BINARY_EXPR(And, "Logical and expression (left and right)")
  BIND_BINARY_EXPR(Or, "Logical or expression (left or right)")
  BIND_BINARY_EXPR(Xor, "Logical xor expression (left xor right)")
  BIND_BINARY_EXPR(BitAnd, "Bitwise and expression (left & right)")
  BIND_BINARY_EXPR(BitOr, "Bitwise or expression (left | right)")
  BIND_BINARY_EXPR(BitXor, "Bitwise xor expression (left ^ right)")
  BIND_BINARY_EXPR(BitShiftLeft, "Bitwise left shift expression (left << right)")
  BIND_BINARY_EXPR(BitShiftRight, "Bitwise right shift expression (left >> right)")

#undef BIND_BINARY_EXPR

// Macro to bind unary expression nodes
#define BIND_UNARY_EXPR(OpName, Description)                                                         \
  py::class_<OpName, UnaryExpr, std::shared_ptr<OpName>>(ir, #OpName, Description)            \
      .def(py::init<const ExprPtr&, DataType, const Span&>(), py::arg("operand"), py::arg("dtype"), \
           py::arg("span"), "Create " Description);

  // Bind all unary expression nodes
  BIND_UNARY_EXPR(Abs, "Absolute value expression (abs(operand))")
  BIND_UNARY_EXPR(Neg, "Negation expression (-operand)")
  BIND_UNARY_EXPR(Not, "Logical not expression (not operand)")
  BIND_UNARY_EXPR(BitNot, "Bitwise not expression (~operand)")
  BIND_UNARY_EXPR(Cast, "Cast expression (cast operand to dtype)")

#undef BIND_UNARY_EXPR

  // Bind structural hash and equality functions
  ir.def("structural_hash", static_cast<uint64_t (*)(const IRNodePtr&, bool)>(&structural_hash),
         py::arg("node"), py::arg("enable_auto_mapping") = false,
         "Compute deterministic structural hash of an IR node (ignores Span). "
         "If enable_auto_mapping=True, variable names are ignored (e.g., x+1 and y+1 hash the same). "
         "If enable_auto_mapping=False (default), different variable objects produce different hashes.");
  ir.def("structural_hash", static_cast<uint64_t (*)(const TypePtr&, bool)>(&structural_hash),
         py::arg("type"), py::arg("enable_auto_mapping") = false,
         "Compute deterministic structural hash of a type. "
         "enable_auto_mapping only affects variables embedded in the type (e.g., shape expressions).");

  ir.def("structural_equal",
         static_cast<bool (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&structural_equal), py::arg("lhs"),
         py::arg("rhs"), py::arg("enable_auto_mapping") = false,
         "Check if two IR nodes are structurally equal. "
         "Ignores source location (Span). Returns True if IR nodes have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");
  ir.def("structural_equal", static_cast<bool (*)(const TypePtr&, const TypePtr&, bool)>(&structural_equal),
         py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false,
         "Check if two types are structurally equal. "
         "Ignores source location (Span). Returns True if types have identical structure. "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  ir.def("assert_structural_equal",
         static_cast<void (*)(const IRNodePtr&, const IRNodePtr&, bool)>(&assert_structural_equal),
         py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false,
         "Assert two IR nodes are structurally equal. "
         "Raises ValueError with detailed error message showing the first mismatch location if they differ. "
         "Ignores source location (Span). "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");
  ir.def("assert_structural_equal",
         static_cast<void (*)(const TypePtr&, const TypePtr&, bool)>(&assert_structural_equal),
         py::arg("lhs"), py::arg("rhs"), py::arg("enable_auto_mapping") = false,
         "Assert two types are structurally equal. "
         "Raises ValueError with detailed error message showing the first mismatch location if they differ. "
         "Ignores source location (Span). "
         "If enable_auto_mapping=True, automatically map variables (e.g., x+1 equals y+1). "
         "If enable_auto_mapping=False (default), variable objects must be exactly the same (not just same "
         "name).");

  // Serialization functions
  ir.def(
      "serialize",
      [](const IRNodePtr& node) {
        auto data = serialization::Serialize(node);
        return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
      },
      py::arg("node"), "Serialize an IR node to MessagePack bytes");

  ir.def(
      "deserialize",
      [](py::bytes data) {
        auto ptr = PyBytes_AS_STRING(data.ptr());
        auto sz = PyBytes_GET_SIZE(data.ptr());
        std::vector<uint8_t> vec(reinterpret_cast<const uint8_t*>(ptr),
                                 reinterpret_cast<const uint8_t*>(ptr) + sz);
        return serialization::Deserialize(vec);
      },
      py::arg("data"), "Deserialize an IR node from MessagePack bytes");

  ir.def("serialize_to_file", &serialization::SerializeToFile, py::arg("node"), py::arg("path"),
         "Serialize an IR node to a file");

  ir.def("deserialize_from_file", &serialization::DeserializeFromFile, py::arg("path"),
         "Deserialize an IR node from a file");

  // ========== Statements ==========

  // Stmt - abstract base, const shared_ptr
  auto stmt_class =
      py::class_<Stmt, IRNode, std::shared_ptr<Stmt>>(ir, "Stmt", "Base class for all statements");
  BindFields<Stmt>(stmt_class);

  // AssignStmt - const shared_ptr
  auto assign_stmt_class =
      py::class_<AssignStmt, Stmt, std::shared_ptr<AssignStmt>>(ir, "AssignStmt",
                                                                       "Assignment statement: var = value");
  assign_stmt_class.def(py::init<const VarPtr&, const ExprPtr&, const Span&>(), py::arg("var"),
                        py::arg("value"), py::arg("span"), "Create an assignment statement");
  BindFields<AssignStmt>(assign_stmt_class);

  // IfStmt - const shared_ptr
  auto if_stmt_class = py::class_<IfStmt, Stmt, std::shared_ptr<IfStmt>>(
      ir, "IfStmt", "Conditional statement: if condition then then_body else else_body");
  if_stmt_class.def(py::init<const ExprPtr&, const StmtPtr&, const std::optional<StmtPtr>&,
                             const std::vector<VarPtr>&, const Span&>(),
                    py::arg("condition"), py::arg("then_body"), py::arg("else_body") = py::none(),
                    py::arg("return_vars"), py::arg("span"),
                    "Create a conditional statement with then and else branches (else_body can be None)");
  BindFields<IfStmt>(if_stmt_class);

  // YieldStmt - const shared_ptr
  auto yield_stmt_class =
      py::class_<YieldStmt, Stmt, std::shared_ptr<YieldStmt>>(ir, "YieldStmt",
                                                                     "Yield statement: yield value");
  yield_stmt_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("value"),
                       py::arg("span"), "Create a yield statement with a list of expressions");
  yield_stmt_class.def(py::init<const Span&>(), py::arg("span"),
                       "Create a yield statement without values");
  BindFields<YieldStmt>(yield_stmt_class);

  // ReturnStmt - const shared_ptr
  auto return_stmt_class =
      py::class_<ReturnStmt, Stmt, std::shared_ptr<ReturnStmt>>(ir, "ReturnStmt",
                                                                       "Return statement: return value");
  return_stmt_class.def(py::init<const std::vector<ExprPtr>&, const Span&>(), py::arg("value"),
                        py::arg("span"), "Create a return statement with a list of expressions");
  return_stmt_class.def(py::init<const Span&>(), py::arg("span"),
                        "Create a return statement without values");
  BindFields<ReturnStmt>(return_stmt_class);

  // ForKind enum (must be before ForStmt which uses it)
  py::enum_<ForKind>(ir, "ForKind", "For loop kind classification")
      .value("Sequential", ForKind::Sequential, "Standard sequential for loop (default)")
      .value("Parallel", ForKind::Parallel, "Parallel for loop")
      .value("Unroll", ForKind::Unroll, "Compile-time unrolled for loop")
      .export_values();

  // ChunkPolicy enum (must be before ForStmt which uses it)
  py::enum_<ChunkPolicy>(ir, "ChunkPolicy", "Chunk policy for loop chunking")
      .value("LeadingFull", ChunkPolicy::LeadingFull, "Full chunks first, smaller remainder at end")
      .export_values();

  // LoopOrigin enum (must be before ForStmt which uses it)
  py::enum_<LoopOrigin>(ir, "LoopOrigin", "Loop origin classification")
      .value("Original", LoopOrigin::Original, "Regular loop (default)")
      .value("ChunkOuter", LoopOrigin::ChunkOuter, "Outer loop from chunk splitting")
      .value("ChunkInner", LoopOrigin::ChunkInner, "Inner loop from chunk splitting")
      .value("ChunkRemainder", LoopOrigin::ChunkRemainder, "Remainder loop from chunk splitting")
      .export_values();

  // ForStmt - const shared_ptr
  auto for_stmt_class = py::class_<ForStmt, Stmt, std::shared_ptr<ForStmt>>(
      ir, "ForStmt", "For loop statement: for loop_var in range(start, stop, step): body");
  for_stmt_class.def(py::init<const VarPtr&, const ExprPtr&, const ExprPtr&, const ExprPtr&,
                              const std::vector<IterArgPtr>&, const StmtPtr&, const std::vector<VarPtr>&,
                              const Span&, ForKind, const std::optional<ExprPtr>&, ChunkPolicy, LoopOrigin>(),
                     py::arg("loop_var"), py::arg("start"), py::arg("stop"), py::arg("step"),
                     py::arg("iter_args"), py::arg("body"), py::arg("return_vars"), py::arg("span"),
                     py::arg("kind") = ForKind::Sequential, py::arg("chunk_size") = py::none(),
                     py::arg("chunk_policy") = ChunkPolicy::LeadingFull,
                     py::arg("loop_origin") = LoopOrigin::Original, "Create a for loop statement");
  BindFields<ForStmt>(for_stmt_class);

  // WhileStmt - const shared_ptr
  auto while_stmt_class =
      py::class_<WhileStmt, Stmt, std::shared_ptr<WhileStmt>>(
          ir, "WhileStmt", "While loop statement: while condition: body");
  while_stmt_class.def(py::init<const ExprPtr&, const std::vector<IterArgPtr>&, const StmtPtr&,
                                const std::vector<VarPtr>&, const Span&>(),
                       py::arg("condition"), py::arg("iter_args"), py::arg("body"), py::arg("return_vars"),
                       py::arg("span"), "Create a while loop statement");
  BindFields<WhileStmt>(while_stmt_class);

  // ScopeKind enum
  py::enum_<ScopeKind>(ir, "ScopeKind", "Scope kind classification")
      .value("InCore", ScopeKind::InCore, "InCore scope for AICore sub-graphs")
      .export_values();

  // ScopeStmt - const shared_ptr
  auto scope_stmt_class = py::class_<ScopeStmt, Stmt, std::shared_ptr<ScopeStmt>>(
      ir, "ScopeStmt", "Scope statement: marks a region with specific execution context");
  scope_stmt_class.def(py::init<ScopeKind, const StmtPtr&, const Span&>(), py::arg("scope_kind"),
                       py::arg("body"), py::arg("span"), "Create a scope statement");
  BindFields<ScopeStmt>(scope_stmt_class);

  // SectionKind enum
  py::enum_<SectionKind>(ir, "SectionKind", "Section kind classification")
      .value("Vector", SectionKind::Vector, "Vector section for vector operations")
      .value("Cube", SectionKind::Cube, "Cube section for cube operations")
      .export_values();

  // SectionStmt - const shared_ptr
  auto section_stmt_class = py::class_<SectionStmt, Stmt, std::shared_ptr<SectionStmt>>(
      ir, "SectionStmt", "Section statement: marks a region with specific section context (Vector or Cube)");
  section_stmt_class.def(py::init<SectionKind, const StmtPtr&, const Span&>(), py::arg("section_kind"),
                         py::arg("body"), py::arg("span"), "Create a section statement");
  BindFields<SectionStmt>(section_stmt_class);

  // SeqStmts - const shared_ptr
  auto seq_stmts_class =
      py::class_<SeqStmts, Stmt, std::shared_ptr<SeqStmts>>(
          ir, "SeqStmts", "Sequence of statements: a sequence of statements");
  seq_stmts_class.def(py::init<const std::vector<StmtPtr>&, const Span&>(), py::arg("stmts"),
                      py::arg("span"), "Create a sequence of statements");
  BindFields<SeqStmts>(seq_stmts_class);

  // OpStmts - const shared_ptr
  auto op_stmts_class = py::class_<OpStmts, Stmt, std::shared_ptr<OpStmts>>(
      ir, "OpStmts", "Operation statements: a sequence of assignment and/or evaluation statements");
  op_stmts_class.def(py::init<const std::vector<StmtPtr>&, const Span&>(), py::arg("stmts"), py::arg("span"),
                     "Create an operation statements");
  BindFields<OpStmts>(op_stmts_class);

  // EvalStmt - const shared_ptr
  auto eval_stmt_class =
      py::class_<EvalStmt, Stmt, std::shared_ptr<EvalStmt>>(ir, "EvalStmt",
                                                                   "Evaluation statement: expr");
  eval_stmt_class.def(py::init<const ExprPtr&, const Span&>(), py::arg("expr"), py::arg("span"),
                      "Create an evaluation statement");
  BindFields<EvalStmt>(eval_stmt_class);

  // BreakStmt - const shared_ptr
  auto break_stmt_class =
      py::class_<BreakStmt, Stmt, std::shared_ptr<BreakStmt>>(ir, "BreakStmt",
                                                                     "Break statement: break");
  break_stmt_class.def(py::init<const Span&>(), py::arg("span"), "Create a break statement");
  BindFields<BreakStmt>(break_stmt_class);

  // ContinueStmt - const shared_ptr
  auto continue_stmt_class =
      py::class_<ContinueStmt, Stmt, std::shared_ptr<ContinueStmt>>(ir, "ContinueStmt",
                                                                           "Continue statement: continue");
  continue_stmt_class.def(py::init<const Span&>(), py::arg("span"), "Create a continue statement");
  BindFields<ContinueStmt>(continue_stmt_class);

  // FunctionType enum
  py::enum_<FunctionType>(ir, "FunctionType", "Function type classification")
      .value("Opaque", FunctionType::Opaque, "Unspecified function type (default)")
      .value("Orchestration", FunctionType::Orchestration, "Host/AICPU control and coordination")
      .value("InCore", FunctionType::InCore, "AICore sub-graph execution")
      .value("Helper", FunctionType::Helper, "Scalar helper callable from kernels (generates func.call)")
      .export_values();

  // ParamDirection enum
  py::enum_<ParamDirection>(ir, "ParamDirection", "Parameter direction classification")
      .value("In", ParamDirection::In, "Read-only input (default)")
      .value("Out", ParamDirection::Out, "Write-only output")
      .value("InOut", ParamDirection::InOut, "Read-write input/output")
      .export_values();

  // Function - const shared_ptr
  auto function_class = py::class_<Function, IRNode, std::shared_ptr<Function>>(
      ir, "Function", "Function definition with name, parameters, return types, and body");
  function_class.def(
      py::init([](const std::string& name, const py::list& params,
                  const std::vector<TypePtr>& return_types, const StmtPtr& body, const Span& span,
                  FunctionType type) -> std::shared_ptr<Function> {
        std::vector<VarPtr> param_vars;
        std::vector<ParamDirection> param_dirs;
        param_vars.reserve(py::len(params));
        param_dirs.reserve(py::len(params));
        for (auto item : params) {
          // Accept either a Var (default In) or a tuple (Var, ParamDirection)
          if (py::isinstance<py::tuple>(item)) {
            auto tup = py::cast<py::tuple>(item);
            if (py::len(tup) != 2) {
              throw pypto::TypeError("Each tuple in 'params' must be (Var, ParamDirection)");
            }
            param_vars.push_back(py::cast<VarPtr>(tup[0]));
            param_dirs.push_back(py::cast<ParamDirection>(tup[1]));
          } else {
            param_vars.push_back(py::cast<VarPtr>(item));
            param_dirs.push_back(ParamDirection::In);
          }
        }
        return std::make_shared<Function>(name, std::move(param_vars), std::move(param_dirs), return_types,
                                         body, span, type);
      }),
      py::arg("name"), py::arg("params"), py::arg("return_types"), py::arg("body"), py::arg("span"),
      py::arg("type") = FunctionType::Opaque, "Create a function definition");
  BindFields<Function>(function_class);

  // Program - const shared_ptr
  auto program_class =
      py::class_<Program, IRNode, std::shared_ptr<Program>>(
          ir, "Program",
          "Program definition with functions mapped by GlobalVar references. "
          "Functions are automatically sorted by name for deterministic ordering.");
  program_class.def(py::init<const std::vector<FunctionPtr>&, const std::string&, const Span&>(),
                    py::arg("functions"), py::arg("name"), py::arg("span"),
                    "Create a program from a list of functions. "
                    "GlobalVar references are created automatically from function names.");
  program_class.def("get_function", &Program::GetFunction, py::arg("name"),
                    "Get a function by name, returns None if not found");
  program_class.def("get_global_var", &Program::GetGlobalVar, py::arg("name"),
                    "Get a GlobalVar by name, returns None if not found");
  // Custom property for functions_ map that converts to Python dict
  program_class.def_property_readonly(
      "functions",
      [](const std::shared_ptr<const Program>& self) {
        py::dict result;
        for (const auto& [gvar, func] : self->functions_) {
          result[py::cast(gvar)] = py::cast(func);
        }
        return result;
      },
      "Map of GlobalVar references to their corresponding functions, sorted by GlobalVar name");
  program_class.def_readonly("name", &Program::name_, "Program name");
  program_class.def_readonly("span", &Program::span_, "Source location");

  // Python-style printer function - unified API for IRNode
  ir.def(
      "python_print",
      [](const IRNodePtr& node, const std::string& prefix) { return PythonPrint(node, prefix); },
      py::arg("node"), py::arg("prefix") = "pl",
      "Print IR node (Expr, Stmt, Function, or Program) in Python IR syntax.\n\n"
      "Args:\n"
      "    node: IR node to print\n"
      "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')");

  // Python-style printer function for Type objects
  ir.def(
      "python_print_type",
      [](const TypePtr& type, const std::string& prefix) { return PythonPrint(type, prefix); },
      py::arg("type"), py::arg("prefix") = "pl",
      "Print Type object in Python IR syntax.\n\n"
      "Args:\n"
      "    type: Type to print\n"
      "    prefix: Module prefix (default 'pl' for 'import pypto.language as pl')");

  // operator functions for Var (wrapped in Python for span capture and normalization)
  ir.def("add", &MakeAdd, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Addition operator");
  ir.def("sub", &MakeSub, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Subtraction operator");
  ir.def("mul", &MakeMul, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Multiplication operator");
  ir.def("truediv", &MakeFloatDiv, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "True division operator");
  ir.def("floordiv", &MakeFloorDiv, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Floor division operator");
  ir.def("mod", &MakeFloorMod, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Modulo operator");
  ir.def("pow", &MakePow, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Power operator");
  ir.def("eq", &MakeEq, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Equality operator");
  ir.def("ne", &MakeNe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Inequality operator");
  ir.def("lt", &MakeLt, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Less than operator");
  ir.def("le", &MakeLe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Less than or equal operator");
  ir.def("gt", &MakeGt, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Greater than operator");
  ir.def("ge", &MakeGe, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Greater than or equal operator");
  ir.def("neg", &MakeNeg, py::arg("operand"), py::arg("span") = Span::unknown(), "Negation operator");
  ir.def("cast", &MakeCast, py::arg("operand"), py::arg("dtype"), py::arg("span") = Span::unknown(),
         "Cast operator");
  ir.def("bit_and", &MakeBitAnd, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Bitwise and operator");
  ir.def("bit_or", &MakeBitOr, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Bitwise or operator");
  ir.def("bit_xor", &MakeBitXor, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Bitwise xor operator");
  ir.def("bit_shift_left", &MakeBitShiftLeft, py::arg("lhs"), py::arg("rhs"),
         py::arg("span") = Span::unknown(), "Bitwise left shift operator");
  ir.def("bit_shift_right", &MakeBitShiftRight, py::arg("lhs"), py::arg("rhs"),
         py::arg("span") = Span::unknown(), "Bitwise right shift operator");
  ir.def("bit_not", &MakeBitNot, py::arg("operand"), py::arg("span") = Span::unknown(),
         "Bitwise not operator");
  ir.def("min_", &MakeMin, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Minimum operator");
  ir.def("max_", &MakeMax, py::arg("lhs"), py::arg("rhs"), py::arg("span") = Span::unknown(),
         "Maximum operator");

  // ParentStmtAnalysis - utility class for analyzing statement parent relationships
  auto parent_stmt_analysis_class = py::class_<ParentStmtAnalysis>(
      ir, "ParentStmtAnalysis",
      "Utility class for analyzing parent-child relationships in statement trees.\n\n"
      "This class builds a mapping from each statement to its parent statement within\n"
      "a function's body. It is useful for passes that need to traverse upward in the\n"
      "IR tree or understand the context of a statement.\n\n"
      "Example usage:\n"
      "    analysis = ir.ParentStmtAnalysis()\n"
      "    analysis.build_map(function)\n"
      "    parent = analysis.get_parent(some_stmt)\n"
      "    if parent:\n"
      "        # Use parent statement\n\n"
      "Note: The analysis becomes invalid after IR transformations. Call build_map again\n"
      "if the IR tree is modified.");

  parent_stmt_analysis_class.def(py::init<>(), "Create a ParentStmtAnalysis instance");

  parent_stmt_analysis_class.def(
      "build_map", &ParentStmtAnalysis::BuildMap, py::arg("func"),
      "Build the parent mapping from a function's body.\n\n"
      "Traverses the function's statement tree and records parent-child relationships.\n"
      "This method clears any existing mapping before building the new one.\n\n"
      "Args:\n"
      "    func: The function to analyze (can be None, resulting in empty map)\n\n"
      "Parent relationships established:\n"
      "- For SeqStmts/OpStmts: Each child statement's parent is the SeqStmts/OpStmts\n"
      "- For IfStmt: then_body and else_body (if present) have IfStmt as parent\n"
      "- For ForStmt: body has ForStmt as parent\n"
      "- Root statement (function.body) has no parent");

  parent_stmt_analysis_class.def("get_parent", &ParentStmtAnalysis::GetParent, py::arg("stmt"),
                                 "Get the parent statement of a given statement.\n\n"
                                 "Args:\n"
                                 "    stmt: The statement to query\n\n"
                                 "Returns:\n"
                                 "    Parent statement, or None if:\n"
                                 "    - stmt is the root statement (function body)\n"
                                 "    - stmt is not found in the analyzed tree\n"
                                 "    - stmt is None");

  parent_stmt_analysis_class.def("has_parent", &ParentStmtAnalysis::HasParent, py::arg("stmt"),
                                 "Check if a statement has a recorded parent.\n\n"
                                 "Args:\n"
                                 "    stmt: The statement to check\n\n"
                                 "Returns:\n"
                                 "    True if stmt has a parent in the map, False otherwise");

  parent_stmt_analysis_class.def("clear", &ParentStmtAnalysis::Clear,
                                 "Clear the parent mapping.\n\n"
                                 "Removes all recorded parent-child relationships. Useful for reusing\n"
                                 "the same ParentStmtAnalysis instance with different functions.");

  // Op conversion registry bindings
  ir.def(
      "register_op_conversion",
      [](const std::string& from_op, const std::string& to_op) {
        OpConversionRegistry::GetInstance().RegisterSimple(from_op, to_op);
      },
      py::arg("from_op"), py::arg("to_op"),
      "Register a simple tensor-to-block op name mapping.\n\n"
      "Args:\n"
      "    from_op: Source op name (e.g., 'tensor.add')\n"
      "    to_op: Target op name (e.g., 'block.add')");

  ir.def(
      "register_op_conversion_custom",
      [](const std::string& from_op, py::object func) {
        // Capture Python callable in a C++ ConversionFunc
        OpConversionRegistry::GetInstance().RegisterCustom(
            from_op,
            [func](const std::vector<ExprPtr>& args,
                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                   const Span& span) -> ConversionResult {
              py::gil_scoped_acquire guard;
              // Convert kwargs to Python list of (key, value) tuples
              py::list py_kwargs_list;
              for (const auto& [key, val] : kwargs) {
                py::object py_val =
                    AnyToPyObject<DataType, MemorySpace, bool, int, std::string, double>(val, key);
                py::tuple pair = py::make_tuple(py::cast(key), py_val);
                py_kwargs_list.append(pair);
              }
              py::object result = func(py::cast(args), py_kwargs_list, py::cast(span));
              // Result can be:
              // 1. An ExprPtr (simple conversion)
              // 2. A tuple of (list[StmtPtr], ExprPtr) (complex conversion)
              if (py::isinstance<py::tuple>(result)) {
                py::tuple result_tuple = py::cast<py::tuple>(result);
                auto prologue = py::cast<std::vector<StmtPtr>>(result_tuple[0]);
                auto expr = py::cast<ExprPtr>(result_tuple[1]);
                return ConversionResult{std::move(prologue), std::move(expr)};
              }
              return ConversionResult{py::cast<ExprPtr>(result)};
            });
      },
      py::arg("from_op"), py::arg("func"),
      "Register a custom conversion function for a tensor op.\n\n"
      "The function receives (args, kwargs, span) and should return either:\n"
      "- An Expr (simple conversion)\n"
      "- A tuple (list[Stmt], Expr) for complex conversions with prologue statements");

  ir.def(
      "has_op_conversion",
      [](const std::string& op_name) { return OpConversionRegistry::GetInstance().HasConversion(op_name); },
      py::arg("op_name"), "Check if a conversion rule exists for an operator.");
}

}  // namespace python
}  // namespace pypto
