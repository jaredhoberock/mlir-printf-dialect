#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::printf {

// @printf's type: (ptr, ...) -> i32
static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
  return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy, /*isVarArg=*/true);
}

// @vprintf's type: (ptr, ptr) -> i32
static LLVM::LLVMFunctionType getVprintfType(MLIRContext *context) {
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
  return LLVM::LLVMFunctionType::get(llvmI32Ty, {llvmPtrTy, llvmPtrTy}, /*isVarArg=*/false);
}

static FlatSymbolRefAttr getOrInsertPrintf(ModuleOp module,
                                           ConversionPatternRewriter &rewriter) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                    getPrintfType(context));
  return SymbolRefAttr::get(context, "printf");
}

static FlatSymbolRefAttr getOrInsertVprintf(gpu::GPUModuleOp gpuModule,
                                            ConversionPatternRewriter &rewriter) {
  auto *context = gpuModule.getContext();
  if (gpuModule.lookupSymbol<LLVM::LLVMFuncOp>("vprintf"))
    return SymbolRefAttr::get(context, "vprintf");

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(gpuModule.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(gpuModule.getLoc(), "vprintf",
                                    getVprintfType(context));
  return SymbolRefAttr::get(context, "vprintf");
}

// Extract arguments, converting memrefs to pointers
static SmallVector<Value> extractArgs(PrintfOp op, PrintfOp::Adaptor adaptor,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc) {
  SmallVector<Value> args;
  for (auto [origArg, convertedArg] : llvm::zip(op.getArgs(), adaptor.getArgs())) {
    if (isa<MemRefType>(origArg.getType())) {
      MemRefDescriptor desc(convertedArg);
      args.push_back(desc.alignedPtr(rewriter, loc));
    } else {
      args.push_back(convertedArg);
    }
  }
  return args;
}

/// Host lowering: printf.printf -> llvm.call @printf
struct PrintfOpHostLowering : OpConversionPattern<PrintfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getParentOfType<gpu::GPUModuleOp>())
      return rewriter.notifyMatchFailure(op, "inside a gpu.module");

    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr printfRef = getOrInsertPrintf(module, rewriter);

    MemRefDescriptor formatDesc(adaptor.getFormat());
    Value formatPtr = formatDesc.alignedPtr(rewriter, loc);

    SmallVector<Value> operands;
    operands.push_back(formatPtr);
    for (Value arg : extractArgs(op, adaptor, rewriter, loc))
      operands.push_back(arg);

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, getPrintfType(rewriter.getContext()), printfRef, operands);
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

/// NVVM lowering: printf.printf -> llvm.call @vprintf
struct PrintfOpNVVMLowering : OpConversionPattern<PrintfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto gpuModule = op->getParentOfType<gpu::GPUModuleOp>();
    if (!gpuModule)
      return rewriter.notifyMatchFailure(op, "not inside a gpu.module");

    auto targets = gpuModule.getTargetsAttr();
    if (!targets || targets.size() != 1 || !isa<NVVM::NVVMTargetAttr>(targets[0]))
      return rewriter.notifyMatchFailure(op, "gpu.module does not unambiguously target NVVM");

    Location loc = op.getLoc();
    auto *context = rewriter.getContext();
    FlatSymbolRefAttr vprintfRef = getOrInsertVprintf(gpuModule, rewriter);

    MemRefDescriptor formatDesc(adaptor.getFormat());
    Value formatPtr = formatDesc.alignedPtr(rewriter, loc);

    SmallVector<Value> args = extractArgs(op, adaptor, rewriter, loc);

    Value argsPtr;
    if (args.empty()) {
      argsPtr = rewriter.create<LLVM::ZeroOp>(loc, LLVM::LLVMPointerType::get(context));
    } else {
      SmallVector<Type> argTypes;
      for (Value arg : args)
        argTypes.push_back(arg.getType());
      auto structTy = LLVM::LLVMStructType::getLiteral(context, argTypes);

      Value one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
      argsPtr = rewriter.create<LLVM::AllocaOp>(
          loc, LLVM::LLVMPointerType::get(context), structTy, one);

      for (auto [i, arg] : llvm::enumerate(args)) {
        Value elemPtr = rewriter.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(context), structTy, argsPtr,
            ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(i)});
        rewriter.create<LLVM::StoreOp>(loc, arg, elemPtr);
      }
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, getVprintfType(context), vprintfRef,
        ValueRange{formatPtr, argsPtr});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

void populatePrintfToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  patterns.add<
    PrintfOpHostLowering,
    PrintfOpNVVMLowering
  >(typeConverter, patterns.getContext());

  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
}

} // end mlir::printf
