#ifndef PRINTF_OPS_TD
#define PRINTF_OPS_TD

include "mlir/IR/OpBase.td"
include "Dialect.td"

def PrintfOp : Op<PrintfDialect, "printf"> {
  let summary = "Calls C printf with a format string and arguments";
  let description = [{
    Emits a call to the C `printf` function. The first operand must be a memref (or pointer)
    to a constant format string. Remaining operands are the arguments to be formatted.

    Example:
      printf.printf %format_string, %arg1, %arg2 : memref<?xi8>, i32, f64
  }];

  let arguments = (ins StrAttr:$format_string, Variadic<AnyType>:$args);
  let results = (outs I32:$result);

  let assemblyFormat = "$format_string (`,` $args^ `:` type($args))? attr-dict `:` type(results)";
}

#endif // PRINTF_OPS_TD
