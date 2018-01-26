// Demonstrates how to list all available frame buffer configurations
// on an X11 display.
#include <assert.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <X11/Xlib.h>
#include <X11/Xresource.h>
#include <X11/Xutil.h>
#include "datatypes/common.h"

void
describe_fbconfig(Display *dpy, GLXFBConfig conf) {
    int xid, r, g, b, a, dbuf, depth, draw_mask, lvl;
    glXGetFBConfigAttrib(dpy, conf, GLX_FBCONFIG_ID, &xid);
    glXGetFBConfigAttrib(dpy, conf, GLX_DRAWABLE_TYPE, &draw_mask);
    glXGetFBConfigAttrib(dpy, conf, GLX_DOUBLEBUFFER, &dbuf);
    glXGetFBConfigAttrib(dpy, conf, GLX_LEVEL, &lvl);
    glXGetFBConfigAttrib(dpy, conf, GLX_RED_SIZE, &r);
    glXGetFBConfigAttrib(dpy, conf, GLX_GREEN_SIZE, &g);
    glXGetFBConfigAttrib(dpy, conf, GLX_BLUE_SIZE, &b);
    glXGetFBConfigAttrib(dpy, conf, GLX_ALPHA_SIZE, &a);
    glXGetFBConfigAttrib(dpy, conf, GLX_DEPTH_SIZE, &depth);
    printf("XID: %d, "
           "Draw. Type: 0x%x, "
           "Doublebuffer: %3s, "
           "Level: %d, "
           "RGBA bits: %d/%d/%d/%d, "
           "Depth: %d\n",
           xid,
           draw_mask,
           dbuf == 1 ? "yes" : "no",
           lvl,
           r, g, b, a,
           depth);
}

int
main(int argc, char *argv[]) {
    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        error("Couldn't connect to X server.");
    }
    int screen = DefaultScreen(dpy);
    int count;
    GLXFBConfig *fbconfigs = glXChooseFBConfig(dpy, screen,
                                               NULL, &count);
    assert(fbconfigs);
    for (int n = 0; n < count; n++) {
        printf("%3d: ", n);
        describe_fbconfig(dpy, fbconfigs[n]);
    }
    XCloseDisplay(dpy);
    return 0;
}
