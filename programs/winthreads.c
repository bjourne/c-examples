// Does windows threads keep running?
#include <stdbool.h>
#include <stdio.h>
#include <windows.h>

static DWORD WINAPI
thread_proc(LPVOID args) {
    while (true) {
        printf("Loop running\n");
        Sleep(10);
    }
}

int
wmain(int argc, wchar_t** argv) {
    printf("Starting thread.\n");
    HANDLE thread = CreateThread(NULL, 0, thread_proc, NULL, 0, NULL);
    SetThreadPriority(thread, THREAD_PRIORITY_ABOVE_NORMAL);
    Sleep(1000);
    exit(10);
    return 0;
}
