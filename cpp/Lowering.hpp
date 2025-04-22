#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace printf {

void populatePrintfToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                            RewritePatternSet& patterns);
}
}
