// RUN: opt -inline %s | FileCheck %s

// MLIR's inliner clones a callee only when every op in it belongs to a dialect
// registering an inliner interface. A function holding a printf op is inlinable
// only because the printf dialect registers one; without it the call below
// survives. After -inline @caller holds the printf.printf directly and no call
// remains.

// CHECK-LABEL: func @caller
// CHECK-NOT: call
// CHECK: printf.printf
// CHECK: return
func.func @caller(%fmt: memref<14xi8>) -> i32 {
  %r = call @callee(%fmt) : (memref<14xi8>) -> i32
  return %r : i32
}

func.func private @callee(%fmt: memref<14xi8>) -> i32 {
  %r = printf.printf(%fmt) : (memref<14xi8>) -> i32
  return %r : i32
}
