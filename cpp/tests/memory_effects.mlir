// RUN: opt --ownership-based-buffer-deallocation %s | FileCheck %s --check-prefix=DEALLOC
// RUN: opt --canonicalize %s | FileCheck %s --check-prefix=KEEP

// printf.printf declares a read and a write of memory with no value. Upstream's
// ownership-based deallocation refuses a function holding an op that declares
// no effects; with the declaration it accepts a function that prints from a
// buffer the function allocated, and frees the buffer after the print.

// DEALLOC-LABEL: func.func @print_from_owned_buffer
// DEALLOC: %[[ALLOC:.*]] = memref.alloc() : memref<4xi8>
// DEALLOC: printf.printf(%[[ALLOC]])
// DEALLOC: %[[BASE:.*]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[ALLOC]]
// DEALLOC: bufferization.dealloc (%[[BASE]] : memref<i8>) if (%true)
// DEALLOC-NEXT: return

// The write keeps a print whose result is unused: canonicalize does not erase it.
// KEEP-LABEL: func.func @print_from_owned_buffer
// KEEP: printf.printf
func.func @print_from_owned_buffer() {
  %fmt = memref.alloc() : memref<4xi8>
  %r = printf.printf(%fmt) : (memref<4xi8>) -> i32
  return
}
