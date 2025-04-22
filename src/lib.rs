use melior::{ir::{Location, Value, Operation}, Context, StringRef};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirStringRef, MlirValue};

#[link(name = "printf_dialect")]
unsafe extern "C" {
    fn printfRegisterDialect(ctx: MlirContext);
    fn printfPrintfOpCreate(loc: MlirLocation, fmtString: MlirStringRef, args: *const MlirValue, numArgs: isize) -> MlirOperation;
}

pub fn register(context: &Context) {
    unsafe { printfRegisterDialect(context.to_raw()) }
}

pub fn printf<'c>(loc: Location<'c>, fmt_str: StringRef, arguments: &[Value<'c, '_>]) -> Operation<'c> {
    let op = unsafe {
        printfPrintfOpCreate(loc.to_raw(), fmt_str.to_raw(), arguments.as_ptr() as *const _, arguments.len() as isize)
    };
    unsafe { Operation::from_raw(op) }
}
