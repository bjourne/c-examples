// Writes and reads llvm bytecode
#include <stdio.h>
#include <stdlib.h>
#include <llvm-c/BitReader.h>
#include <llvm-c/BitWriter.h>
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>

// static LLVMValueRef
// build_sum_function(LLVMModuleRef mod) {
//     LLVMTypeRef param_types[] = { LLVMInt32Type(), LLVMInt32Type() };
//     LLVMTypeRef ret_type = LLVMFunctionType(LLVMInt32Type(),
//                                             param_types,
//                                             2,
//                                             0);
//     LLVMValueRef sum = LLVMAddFunction(mod, "sum", ret_type);
//     LLVMBasicBlockRef entry = LLVMAppendBasicBlock(sum, "entry");
//     LLVMBuilderRef builder = LLVMCreateBuilder();
//     LLVMPositionBuilderAtEnd(builder, entry);
//     LLVMValueRef tmp = LLVMBuildAdd(builder,
//                                     LLVMGetParam(sum, 0),
//                                     LLVMGetParam(sum, 1),
//                                     "tmp");
//     LLVMBuildRet(builder, tmp);
//     LLVMDisposeBuilder(builder);
//     return sum;
// }


int
main(int argc, char* argv[]) {
    //LLVMModuleRef mod = LLVMModuleCreateWithName("my_module");
    // build_sum_function(mod);

    // if (LLVMWriteBitcodeToFile(mod, "sum.bc")) {
    //     printf("Error writing bitcode to file, skipping\n");
    //     return 1;
    // }

    char *error = NULL;
    LLVMMemoryBufferRef membuf;
    if (LLVMCreateMemoryBufferWithContentsOfFile("sum.bc", &membuf, &error)) {
        printf("Failed to create memory buffer: %s\n", error);
        LLVMDisposeMessage(error);
        return 1;
    }
    LLVMModuleRef mod;
    if (LLVMParseBitcode(membuf, &mod, &error))  {
        printf("Failed to parse bitcode: %s\n", error);
        LLVMDisposeMessage(error);
        return 1;
    }
    LLVMLinkInMCJIT();
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
    LLVMExecutionEngineRef engine;
    if (LLVMCreateExecutionEngineForModule(&engine, mod, &error)) {
        printf("Failed to create execution engine: %s\n", error);
        LLVMDisposeMessage(error);
        return 1;
    }
    LLVMValueRef sum = LLVMGetFirstFunction(mod);
    int32_t x = strtoll(argv[1], NULL, 10);
    int32_t y = strtoll(argv[2], NULL, 10);
    int32_t (*funcPtr) (int32_t, int32_t)
        = (int32_t (*)(int32_t, int32_t)) LLVMGetPointerToGlobal(engine, sum);
    // Run the code!
    printf("%d\n", funcPtr(x, y));
    return 0;
}
