#ifndef PRINTF_OPS_TD
#define PRINTF_OPS_TD

include "mlir/IR/OpBase.td"
include "Dialect.td"

def PrintfOp : Op<PrintfDialect, "printf"> {
  let summary = "Calls C printf with a format string and arguments";

  let description = [{
    Calls the C `printf` function. The `format` operand must be a memref pointing to a null-terminated string of i8s.
    The remaining operands are variadic and correspond to format string arguments.

    Example:
      printf.printf(%format, %x) : (memref<?xi8>, i32) -> i32
  }];

  let arguments = (ins MemRefOf<[I8]>:$format, Variadic<AnyType>:$args);
  let results = (outs I32:$result);

  let assemblyFormat = "`(` $format (`,` $args^)? `)` attr-dict `:` functional-type(operands, results)";
}

#endif // PRINTF_OPS_TD
