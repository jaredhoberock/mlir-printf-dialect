#include "Dialect.hpp"
#include "Ops.hpp"
#include "Lowering.hpp"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include <Dialect.cpp.inc>

namespace mlir::printf {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populatePrintfToLLVMConversionPatterns(typeConverter, patterns);
  }
};

/// Declares inlining legal for the printf dialect. MLIR's inliner clones a callee
/// only when every op in it belongs to a dialect registering this interface, so a
/// function holding a printf op is inlinable only once printf provides one. The
/// hooks are permissive: printf's ops carry no state that forbids cloning them
/// into another region.
struct PrintfInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

void PrintfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <Ops.cpp.inc>
  >();

  addInterfaces<
    ConvertToLLVMInterface,
    PrintfInlinerInterface
  >();
}

}
