// RUN: opt --convert-to-llvm %s | FileCheck %s

// ---- A device printf lowered from a non-entry block allocates its vprintf
// argument buffer in the function entry block. The alloca must precede the
// branch; behind it the buffer would be a dynamic alloca that bars the kernel
// from being inlined. The buffer is filled at the point of use, so the store
// and the call stay in the branched block.
// CHECK-LABEL: llvm.func @print_behind_branch
// CHECK: llvm.alloca
// CHECK: llvm.cond_br
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @vprintf
// CHECK-NOT: llvm.alloca
gpu.module @kernels [#nvvm.target<chip = "sm_80">] {
  memref.global constant @fmt : memref<4xi8> = dense<[37, 100, 10, 0]>
  func.func @print_behind_branch(%cond: i1, %x: i32) {
    %fmt = memref.get_global @fmt : memref<4xi8>
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %r = printf.printf(%fmt, %x) : (memref<4xi8>, i32) -> i32
    return
  ^bb2:
    return
  }
}
