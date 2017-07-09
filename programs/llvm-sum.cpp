// This code comes from:
//
//  https://pauladamsmith.com/blog/2015/01/how-to-get-started-with-llvm-c-api.html
//
// But fixed to work with LLVM 3.8
#include <stdio.h>
#include <stdlib.h>
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Analysis.h>

int
main(int argc, char* argv[]) {
    LLVMModuleRef mod = LLVMModuleCreateWithName("my_module");

    LLVMTypeRef param_types[] = { LLVMInt32Type(), LLVMInt32Type() };
    LLVMTypeRef ret_type = LLVMFunctionType(LLVMInt32Type(),
                                            param_types,
                                            2,
                                            0);
    LLVMValueRef sum = LLVMAddFunction(mod, "sum", ret_type);
    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(sum, "entry");
    LLVMBuilderRef builder = LLVMCreateBuilder();
    LLVMPositionBuilderAtEnd(builder, entry);
    LLVMValueRef tmp = LLVMBuildAdd(builder,
                                    LLVMGetParam(sum, 0),
                                    LLVMGetParam(sum, 1),
                                    "tmp");
    LLVMBuildRet(builder, tmp);
    LLVMDisposeBuilder(builder);

    char *error = NULL;
    LLVMVerifyModule(mod, LLVMAbortProcessAction, &error);
    LLVMDisposeMessage(error);

    LLVMExecutionEngineRef engine;
    error = NULL;
    LLVMLinkInMCJIT();
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
    if (LLVMCreateExecutionEngineForModule(&engine, mod, &error) != 0) {
        fprintf(stderr, "failed to create execution engine\n");
        fprintf(stderr, "error: %s\n", error);
        LLVMDisposeMessage(error);
        return 1;
    }
    int32_t x = strtoll(argv[1], NULL, 10);
    int32_t y = strtoll(argv[2], NULL, 10);
    int32_t (*funcPtr) (int32_t, int32_t)
        = (int32_t (*)(int32_t, int32_t)) LLVMGetPointerToGlobal(engine, sum);
    // Run the code!
    printf("%d\n", funcPtr(x, y));
    LLVMDumpModule(mod);
    LLVMDisposeExecutionEngine(engine);
    return 0;
}
