memref.global constant @hello_world_fmt : memref<14xi8> = dense<[72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 10]>

func.func @hello_world() {
  %fmt = memref.get_global @hello_world_fmt : memref<14xi8>
  %result = printf.printf(%fmt) : (memref<14xi8>) -> i32
  return
}
