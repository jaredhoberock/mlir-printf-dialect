#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::printf;

extern "C" {

void printfRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<PrintfDialect>();
}

MlirOperation printfPrintfOpCreate(MlirLocation loc, MlirValue format,
                                   MlirValue *varArgs, intptr_t numVarArgs) {
  MLIRContext *context = unwrap(loc)->getContext();
  OpBuilder builder(context);

  SmallVector<Value> operands;
  operands.push_back(unwrap(format));
  for (intptr_t i = 0; i < numVarArgs; ++i)
    operands.push_back(unwrap(varArgs[i]));

  auto resultType = builder.getI32Type();

  auto printfOp = builder.create<printf::PrintfOp>(
      unwrap(loc), resultType, operands);

  return wrap(printfOp.getOperation());
}

} // end extern "C"
