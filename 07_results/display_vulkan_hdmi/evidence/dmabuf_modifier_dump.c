// labwc 가 zwp_linux_dmabuf_v1 로 광고하는 format/modifier 표를 덤프한다.
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <wayland-client.h>
#include "dmabuf-client.h"

static struct zwp_linux_dmabuf_v1 *dmabuf;
static int n_mod = 0, sand_nv12 = 0, linear_nv12 = 0;

static void fourcc_str(uint32_t f, char *o) {
    for (int i = 0; i < 4; i++) { char c = (char)((f >> (8*i)) & 0xff); o[i] = (c >= 32 && c < 127) ? c : '?'; }
    o[4] = 0;
}
static void on_format(void *d, struct zwp_linux_dmabuf_v1 *p, uint32_t fmt) { (void)d;(void)p;(void)fmt; }
static void on_modifier(void *d, struct zwp_linux_dmabuf_v1 *p, uint32_t fmt, uint32_t hi, uint32_t lo) {
    (void)d; (void)p;
    uint64_t m = ((uint64_t)hi << 32) | lo;
    char s[5]; fourcc_str(fmt, s);
    if (strcmp(s, "NV12") == 0) {
        printf("  NV12  modifier 0x%016llx\n", (unsigned long long)m);
        if (m == 0x0700000000000004ull) sand_nv12 = 1;
        if (m == 0ull) linear_nv12 = 1;
    }
    n_mod++;
}
static const struct zwp_linux_dmabuf_v1_listener dl = { on_format, on_modifier };

static void on_global(void *d, struct wl_registry *r, uint32_t name, const char *iface, uint32_t ver) {
    (void)d;
    if (strcmp(iface, "zwp_linux_dmabuf_v1") == 0) {
        printf("zwp_linux_dmabuf_v1 광고됨 (컴포지터 version=%u) -> version 3 으로 bind\n", ver);
        uint32_t use = ver < 3 ? ver : 3;
        dmabuf = wl_registry_bind(r, name, &zwp_linux_dmabuf_v1_interface, use);
        zwp_linux_dmabuf_v1_add_listener(dmabuf, &dl, NULL);
    } else if (strcmp(iface, "wp_viewporter") == 0 || strcmp(iface, "wp_presentation") == 0
            || strcmp(iface, "wl_subcompositor") == 0) {
        printf("%-22s 광고됨 (version=%u)\n", iface, ver);
    }
}
static void on_global_remove(void *d, struct wl_registry *r, uint32_t n) { (void)d;(void)r;(void)n; }
static const struct wl_registry_listener rl = { on_global, on_global_remove };

int main(void) {
    struct wl_display *dpy = wl_display_connect(NULL);
    if (!dpy) { fprintf(stderr, "wl_display_connect 실패\n"); return 2; }
    struct wl_registry *reg = wl_display_get_registry(dpy);
    wl_registry_add_listener(reg, &rl, NULL);
    wl_display_roundtrip(dpy);   // globals
    wl_display_roundtrip(dpy);   // format/modifier events
    printf("\nNV12 modifier 총 %d개 이벤트 중\n", n_mod);
    printf("  SAND128(0x0700000000000004) 지원: %s\n", sand_nv12 ? "예" : "아니오");
    printf("  LINEAR(0x0)             지원: %s\n", linear_nv12 ? "예" : "아니오");
    wl_display_disconnect(dpy);
    return sand_nv12 ? 0 : 1;
}
