import os
import lit.formats

config.name = "Printf Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

mlir_prefix = os.environ.get('MLIR_SYS_220_PREFIX', '/home/jhoberock/dev/git/llvm-project-22/build')
llvm_bin_dir = os.path.join(mlir_prefix, 'bin')
filecheck = os.path.join(llvm_bin_dir, 'FileCheck')
if not os.path.exists(filecheck):
    filecheck = '/home/jhoberock/dev/git/llvm-project-22/build-release-asserts/bin/FileCheck'
if not os.path.exists(filecheck):
    filecheck = '/home/jhoberock/dev/git/llvm-project-22/build/bin/FileCheck'
plugin_path = os.environ.get(
    'PRINTF_DIALECT_PLUGIN',
    os.path.join(os.path.dirname(__file__), '..', 'build', 'libprintf_dialect.so'),
)

config.substitutions.append(('opt', f'{os.path.join(llvm_bin_dir, "mlir-opt")} --load-dialect-plugin={plugin_path}'))
config.substitutions.append(('FileCheck', filecheck))
