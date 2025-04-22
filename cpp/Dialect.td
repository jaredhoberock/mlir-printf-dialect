#ifndef PRINTF_DIALECT_TD
#define PRINTF_DIALECT_TD

include "mlir/IR/OpBase.td"

def PrintfDialect : Dialect {
  let name = "printf";
  let summary = "A dialect for calling printf";
  let cppNamespace = "::mlir::printf";

  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

#endif // PRINTF_DIALECT_TD
