#include "Dialect.hpp"
#include "Ops.hpp"
#include "Lowering.hpp"
#include "PermissiveInlinerInterface.hpp"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include <Dialect.cpp.inc>

namespace mlir::printf {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  ConvertToLLVMInterface(mlir::Dialect *dialect)
      : mlir::ConvertToLLVMPatternInterface(dialect) {}

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populatePrintfToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void PrintfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <Ops.cpp.inc>
  >();

  addInterfaces<
    ConvertToLLVMInterface,
    lowering::PermissiveInlinerInterface<PrintfDialect>
  >();
}

}
