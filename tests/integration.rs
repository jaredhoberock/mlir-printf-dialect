use melior::{
    Context,
    dialect::{func, memref, DialectRegistry},
    ExecutionEngine,
    ir::{
        attribute::{AttributeLike, DenseElementsAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        Attribute,
        r#type::{FunctionType, IntegerType, MemRefType, RankedTensorType},
        Block, BlockLike, Location, Module, Region, RegionLike,
    },
    pass::{self, PassManager},
    utility::{register_all_dialects},
};
use printf_dialect as printf;


use std::io::{Read, Seek, SeekFrom};
use std::os::unix::io::AsRawFd;
use libc::{dup, dup2, fflush, fileno};

unsafe extern "C" {
    static mut stdout: *mut libc::FILE;
}

fn capture_stdout<F: FnOnce() -> T, T>(f: F) -> (T, String) {
    unsafe {
        fflush(stdout);
        let old_fd = dup(fileno(stdout));
        let mut tmpfile = tempfile::tempfile().expect("create temp file");
        let tmp_fd = tmpfile.as_raw_fd();

        dup2(tmp_fd, fileno(stdout));

        let result = f();

        fflush(stdout);
        dup2(old_fd, fileno(stdout));
        libc::close(old_fd);

        tmpfile.seek(SeekFrom::Start(0)).unwrap();
        let mut output = String::new();
        tmpfile.read_to_string(&mut output).unwrap();

        (result, output)
    }
}


#[test]
fn test_printf_jit() {
    // create a dialect registry and register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    printf::register(&context);

    // make all the dialects available
    context.load_all_available_dialects();

    // begin creating a module
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    // create a global string "Hello, world!"
    let hello_world_str = b"Hello, world!\n\0";
    let i8_ty = IntegerType::new(&context, 8);
    let memref_ty = MemRefType::new(i8_ty.into(), &[hello_world_str.len() as i64], None, None);
    assert_eq!(memref_ty.to_string(), "memref<15xi8>"); 

    // Build dense attribute
    let values: Vec<_> = hello_world_str
        .iter()
        .map(|b| IntegerAttribute::new(i8_ty.into(), (*b).into()).into())
        .collect();
    
    let tensor_ty = RankedTensorType::new(&[hello_world_str.len() as u64], i8_ty.into(), None);
    let value_attr = DenseElementsAttribute::new(tensor_ty.into(), &values)
        .expect("valid dense string literal");
    assert_eq!(value_attr.r#type().to_string(), "tensor<15xi8>");
    
    // Create the global string constant
    module.body().append_operation(memref::global(
        &context,
        "hello_world_fmt",
        None,                  // no visibility → private
        memref_ty,
        Some(value_attr.into()),
        true,                  // constant
        None,                  // no alignment
        location,
    ));

    let i32_ty = IntegerType::new(&context, 32);
    let function_type = FunctionType::new(
        &context, 
        &[], 
        &[i32_ty.into()]
    );

    // Build the function body:
    //     %str = memref.get_global @hello_world_fmt ...
    //     %r = printf.printf(%str) : (memref<...>) -> i32
    //     return %r : i32
    let region = {
        let block = Block::new(&[]);
        
        let format_ptr = block.append_operation(memref::get_global(
            &context,
            "hello_world_fmt",
            memref_ty,
            location,
        )).result(0).unwrap();

        let printf_result = block.append_operation(printf::printf(
            location,
            &format_ptr.into(),
            &[],
        )).result(0).unwrap();

        block.append_operation(func::r#return(&[printf_result.into()], location));

        let region = Region::new();
        region.append_block(block);
        region
    };

    // Define the function
    let mut func_op = func::func(
        &context,
        StringAttribute::new(&context, "run"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );

    // this attribute tells MLIR to create an additional wrapper function that we can use 
    // to invoke "run" via invoke_packed below
    func_op.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));

    module.body().append_operation(func_op);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false);

    // test that we can call the function and it produces the expected result
    let mut result = [0i32];
    let mut args: [*mut (); 1] = [result.as_mut_ptr().cast()];

    let ((), output) = capture_stdout(|| {
        unsafe {
            engine
                .invoke_packed("run", &mut args)
                .expect("JIT invocation failed");
        }
    });
    
    assert_eq!(output, "Hello, world!\n");
    assert_eq!(result[0], b"Hello, world!\n".len() as i32);
}
