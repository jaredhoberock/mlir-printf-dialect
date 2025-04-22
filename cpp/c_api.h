#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the printf dialect with a context.
void printfRegisterDialect(MlirContext ctx);

/// Creates a printf.printf operation.
MlirOperation printfBuildPrintfOp(MlirLocation loc, MlirStringRef fmtString,
                                  MlirValue *args, intptr_t numArgs);

#ifdef __cplusplus
}
#endif
