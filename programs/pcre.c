// gcc -o pcre pcre.c -lpcre
#include <pcre.h>
#include <stdio.h>

#define PRINT_OPT(o, x)    printf("%-25s: %s\n", #x, (o & x) ? "yes" : "no");

int
main(int argc, char *argv[]) {
    const char *err_str;
    int err_ofs, opts;

    int def_opts = PCRE_UTF8 | PCRE_UCP;

    pcre *c = pcre_compile("(?im)abc", def_opts, &err_str, &err_ofs, NULL);
    if (pcre_fullinfo(c, NULL, PCRE_INFO_OPTIONS, &opts)) {
        printf("Error getting fullinfo.\n");
        return 1;
    }
    printf("=== Regex options ===\n");
    PRINT_OPT(opts, PCRE_ANCHORED);
    PRINT_OPT(opts, PCRE_AUTO_CALLOUT);
    PRINT_OPT(opts, PCRE_BSR_ANYCRLF);
    PRINT_OPT(opts, PCRE_BSR_UNICODE);
    PRINT_OPT(opts, PCRE_CASELESS);
    PRINT_OPT(opts, PCRE_DOLLAR_ENDONLY);
    PRINT_OPT(opts, PCRE_DOTALL);
    PRINT_OPT(opts, PCRE_DUPNAMES);
    PRINT_OPT(opts, PCRE_EXTENDED);
    PRINT_OPT(opts, PCRE_EXTRA);
    PRINT_OPT(opts, PCRE_FIRSTLINE);
    PRINT_OPT(opts, PCRE_JAVASCRIPT_COMPAT);
    PRINT_OPT(opts, PCRE_MULTILINE);
    PRINT_OPT(opts, PCRE_NEVER_UTF);
    PRINT_OPT(opts, PCRE_NEWLINE_ANY);
    PRINT_OPT(opts, PCRE_NEWLINE_ANYCRLF);
    PRINT_OPT(opts, PCRE_NEWLINE_CR);
    PRINT_OPT(opts, PCRE_NEWLINE_CRLF);
    PRINT_OPT(opts, PCRE_NEWLINE_LF);
    PRINT_OPT(opts, PCRE_NO_AUTO_CAPTURE);
    PRINT_OPT(opts, PCRE_NO_AUTO_POSSESS);
    PRINT_OPT(opts, PCRE_NO_START_OPTIMIZE);
    PRINT_OPT(opts, PCRE_NO_UTF16_CHECK);
    PRINT_OPT(opts, PCRE_NO_UTF32_CHECK);
    PRINT_OPT(opts, PCRE_NO_UTF8_CHECK);
    PRINT_OPT(opts, PCRE_UCP);
    PRINT_OPT(opts, PCRE_UNGREEDY);
    PRINT_OPT(opts, PCRE_UTF16);
    PRINT_OPT(opts, PCRE_UTF32);
    PRINT_OPT(opts, PCRE_UTF8);

    pcre_free(c);
    return 0;
}
