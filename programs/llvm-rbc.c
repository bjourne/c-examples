// Writes and reads llvm bytecode
#include <stdio.h>
#include <stdlib.h>
#include <llvm-c/BitReader.h>
#include <llvm-c/BitWriter.h>
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>

LLVMModuleRef
load_module (const char *path, char **error) {
    LLVMMemoryBufferRef membuf;
    if (LLVMCreateMemoryBufferWithContentsOfFile(path, &membuf, error)) {
        return NULL;
    }
    LLVMModuleRef mod;
    if (LLVMParseBitcode(membuf, &mod, error))  {
        return NULL;
    }
    LLVMDisposeMemoryBuffer(membuf);
    return mod;
}

int
main(int argc, char* argv[]) {
    char *error = NULL;
    const char *path = "sum.bc";
    LLVMModuleRef mod = load_module(path, &error);
    if (!mod) {
        printf("Failed to load module '%s': %s\n", path, error);
        LLVMDisposeMessage(error);
        return 1;
    }
    LLVMLinkInMCJIT();
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    //LLVMInitializeNativeAsmParser();
    printf("is here\n");
    LLVMExecutionEngineRef engine;
    if (LLVMCreateExecutionEngineForModule(&engine, mod, &error)) {
        printf("Failed to create execution engine: %s\n", error);
        LLVMDisposeMessage(error);
        return 1;
    }
    printf("here?\n");
    LLVMValueRef sum = LLVMGetFirstFunction(mod);
    printf("sum %p\n", sum);
    int32_t x = strtoll(argv[1], NULL, 10);
    int32_t y = strtoll(argv[2], NULL, 10);
    int32_t (*funcPtr) (int32_t, int32_t)
        = (int32_t (*)(int32_t, int32_t)) LLVMGetPointerToGlobal(engine, sum);
    // Run the code!
    printf("running\n");
    printf("%d\n", funcPtr(x, y));
    printf("done\n");
    LLVMDisposeExecutionEngine(engine);
    return 0;
}
