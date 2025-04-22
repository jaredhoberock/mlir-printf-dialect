use melior::{ir::{Location, Value, ValueLike, Operation}, Context};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirValue};

#[link(name = "printf_dialect")]
unsafe extern "C" {
    fn printfRegisterDialect(ctx: MlirContext);
    fn printfPrintfOpCreate(loc: MlirLocation, format: MlirValue, varArgs: *const MlirValue, numVarArgs: isize) -> MlirOperation;
}

pub fn register(context: &Context) {
    unsafe { printfRegisterDialect(context.to_raw()) }
}

pub fn printf<'c>(loc: Location<'c>, format: &Value<'c,'_>, var_args: &[Value<'c, '_>]) -> Operation<'c> {
    let op = unsafe {
        printfPrintfOpCreate(loc.to_raw(), format.to_raw(), var_args.as_ptr() as *const _, var_args.len() as isize)
    };
    unsafe { Operation::from_raw(op) }
}
