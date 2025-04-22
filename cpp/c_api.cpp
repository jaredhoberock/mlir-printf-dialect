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

MlirOperation printfBuildPrintfOp(MlirLocation loc, MlirStringRef fmtString,
                                  MlirValue *args, intptr_t numArgs) {
  MLIRContext *context = unwrap(loc)->getContext();
  OpBuilder builder(context);

  // Convert C strings to MLIR attributes and values
  auto fmtAttr = builder.getStringAttr(
      StringRef(fmtString.data, fmtString.length));

  SmallVector<Value> operands;
  for (intptr_t i = 0; i < numArgs; ++i)
    operands.push_back(unwrap(args[i]));

  auto resultType = builder.getI32Type();

  auto printfOp = builder.create<printf::PrintfOp>(
      unwrap(loc), resultType, fmtAttr, operands);

  return wrap(printfOp.getOperation());
}

} // end extern "C"
