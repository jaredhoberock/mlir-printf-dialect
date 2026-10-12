// RUN: opt --buffer-deallocation-pipeline %s | FileCheck %s --check-prefix=FREE
// RUN: opt --buffer-deallocation-pipeline --mlir-print-op-generic %s | FileCheck %s --check-prefix=FREE
// RUN: opt --canonicalize %s | FileCheck %s --check-prefix=KEEP

// printf.printf declares a read and a write of memory with no value. Upstream's
// ownership-based deallocation refuses a function holding an op that declares
// no effects; with the declaration the deallocation pipeline accepts a function
// that prints from a buffer it allocated, and frees the buffer once, after the
// print. The checks name operations only, so they hold under the generic
// printer too.

// FREE-LABEL: print_from_owned_buffer
// FREE-NOT: memref.dealloc
// FREE: printf.printf
// FREE-COUNT-1: memref.dealloc
// FREE-NOT: memref.dealloc

// The write keeps a print whose result is unused: canonicalize does not erase it.
// KEEP-LABEL: print_from_owned_buffer
// KEEP: printf.printf
func.func @print_from_owned_buffer() {
  %fmt = memref.alloc() : memref<1xi8>
  %c0 = arith.constant 0 : index
  %nul = arith.constant 0 : i8
  memref.store %nul, %fmt[%c0] : memref<1xi8>
  %r = printf.printf(%fmt) : (memref<1xi8>) -> i32
  return
}
