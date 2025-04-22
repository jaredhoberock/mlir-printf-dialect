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
MlirOperation printfPrintfOpCreate(MlirLocation loc, MlirValue format,
                                   MlirValue *varArgs, intptr_t numVarArgs);

#ifdef __cplusplus
}
#endif
