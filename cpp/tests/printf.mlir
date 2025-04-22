// RUN: opt %s | FileCheck %s

// ---- Test 1: No varargs

// CHECK-LABEL: func @hello_world
// CHECK: %[[PRINTF:.*]] = printf.printf
func.func @hello_world() -> () {
  printf.printf "Hello, world!" : i32
  return
}

// ---- Test 2: With varargs

// CHECK-LABEL: func @print_i32
// CHECK: %[[PRINTF:.*]] = printf.printf
func.func @print_i32(%x: i32) -> () {
  printf.printf "number: %d\n", %x : i32 : i32
  return
}
