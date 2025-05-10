#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::printf {

// Create a function declaration for printf, the signature is:
//   * `i32 (i8*, ...)`
static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy, /*isVarArg=*/true);
  return llvmFnType;
}

static FlatSymbolRefAttr getOrInsertPrintf(ModuleOp module, 
                                           ConversionPatternRewriter &rewriter) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                    getPrintfType(context));
  return SymbolRefAttr::get(context, "printf");
}


struct PrintfOpLowering : OpConversionPattern<PrintfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
  
    // get or insert the declaration of `printf`.
    FlatSymbolRefAttr printfRef = getOrInsertPrintf(module, rewriter);

    // extract the aligned pointer of the foramt string from memref descriptor
    MemRefDescriptor formatDesc(adaptor.getFormat());
    Value formatPtr = formatDesc.alignedPtr(rewriter, loc);
  
    // Build the operands list: first the format string, then any other args
    SmallVector<Value> operands;
    operands.push_back(formatPtr);
    operands.append(adaptor.getArgs().begin(), adaptor.getArgs().end());
  
    // Emit the call
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc,
        getPrintfType(rewriter.getContext()),
        printfRef,
        operands);
  
    // Replace the original op
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

void populatePrintfToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  patterns.add<PrintfOpLowering>(typeConverter, patterns.getContext());

  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
}

}
