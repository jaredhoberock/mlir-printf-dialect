#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
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


static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getIndexAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}


struct PrintfOpLowering : OpConversionPattern<PrintfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
  
    // Get or insert the declaration of `printf`.
    FlatSymbolRefAttr printfRef = getOrInsertPrintf(module, rewriter);
  
    // Convert the format string attribute into a global constant.
    StringRef fmtStr = op.getFormatString();
    Value formatStrPtr = getOrCreateGlobalString(loc, rewriter, "format_str", fmtStr, module);
  
    // Build the operands list: first the format string, then any other args
    SmallVector<Value> operands;
    operands.push_back(formatStrPtr);
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
}

}
