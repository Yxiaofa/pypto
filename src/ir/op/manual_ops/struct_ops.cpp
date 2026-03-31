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

/**
 * @file manual_ops/struct_ops.cpp
 * @brief Struct array operations for clean C++ codegen.
 *
 * These ops translate pl.struct arrays with dynamic indexing into
 * C++ struct definitions + array declarations + direct subscript access.
 *
 *   - struct.declare: emit struct type + array declaration
 *   - struct.get:     read  arr[idx].field
 *   - struct.set:     write arr[idx].field = val
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// struct.declare: () → void  (side effect: C++ struct + array declaration)
REGISTER_OP("struct.declare")
    .set_op_category("StructOp")
    .set_description("Declare a C++ struct type and array from pl.struct list.")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& /*args*/,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) -> TypePtr {
      // No return type — used as EvalStmt (side effect)
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

// struct.get: (index) → int64  (returns field value)
REGISTER_OP("struct.get")
    .set_op_category("StructOp")
    .set_description("Read a field from a struct array at a dynamic index.")
    .add_argument("index", "Array index expression")
    .f_deduce_type([](const std::vector<ExprPtr>& /*args*/,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) -> TypePtr {
      // Struct fields are int64_t scalars
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

// struct.set: (index, value) → void  (side effect: write to array)
REGISTER_OP("struct.set")
    .set_op_category("StructOp")
    .set_description("Write a field in a struct array at a dynamic index.")
    .add_argument("index", "Array index expression")
    .add_argument("value", "Value to write")
    .f_deduce_type([](const std::vector<ExprPtr>& /*args*/,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) -> TypePtr {
      // No meaningful return type — used as EvalStmt (side effect)
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

// struct.ref: (index) → void  (side effect: C++ reference declaration)
REGISTER_OP("struct.ref")
    .set_op_category("StructOp")
    .set_description("Declare a C++ reference to a struct array element: auto& var = arr[idx];")
    .add_argument("index", "Array index expression")
    .f_deduce_type([](const std::vector<ExprPtr>& /*args*/,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) -> TypePtr {
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

}  // namespace ir
}  // namespace pypto
