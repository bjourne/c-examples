// This example shows you how to trap and resume after seg faults.
//
// gcc -o sigsegv sigsegv.c
#include <assert.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ucontext.h>

static void
handler(int signal, siginfo_t* siginfo, void* uap) {
    printf("Attempt to access memory at address %p\n", siginfo->si_addr);
    mcontext_t *mctx = &((ucontext_t *)uap)->uc_mcontext;
    greg_t *rsp = &mctx->gregs[15];
    greg_t *rip = &mctx->gregs[16];

    // Jump past the bad memory write.
    *rip = *rip + 7;
    assert(rsp);
}

static void
dobad(uintptr_t *addr) {
    *addr = 0x998877;
    printf("I'm a survivor!\n");
}

int main(int argc, char *argv[]) {
    struct sigaction act;
    memset(&act, 0, sizeof(struct sigaction));
    sigemptyset(&act.sa_mask);
    act.sa_sigaction = handler;
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;

    sigaction(SIGSEGV, &act, NULL);

    // Write to an address we don't have access to.
    dobad((uintptr_t*)0x1234);

    return 0;
}
