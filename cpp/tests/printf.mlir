// RUN: opt %s | FileCheck %s

// ---- Test 1: No varargs

// CHECK-LABEL: func @hello_world
// CHECK: %[[PRINTF:.*]] = printf.printf
memref.global constant @hello_world_fmt : memref<14xi8> = dense<[72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 10]>

func.func @hello_world() {
  %fmt = memref.get_global @hello_world_fmt : memref<14xi8>
  %result = printf.printf(%fmt) : (memref<14xi8>) -> i32
  return
}

// ---- Test 2: With varargs

// CHECK-LABEL: func @print_i32
// CHECK: %[[PRINTF:.*]] = printf.printf
memref.global constant @print_i32_fmt : memref<12xi8> = dense<[110, 117, 109, 98, 101, 114, 58, 32, 37, 100, 10, 0]>

func.func @print_i32(%x: i32) {
  %fmt = memref.get_global @print_i32_fmt : memref<12xi8>
  %result = printf.printf(%fmt, %x) : (memref<12xi8>, i32) -> i32
  return
}
