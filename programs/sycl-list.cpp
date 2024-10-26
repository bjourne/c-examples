// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// This program uses SYCL to list OpenCL devices on the host. To
// compile it, ensure that oneAPI has been activated with something
// like: source ~/intel/oneapi/setvars.sh
#include <CL/sycl.hpp>
#include <map>
#include <vector>

#include <stdio.h>

extern "C" {
    #include "datatypes/common.h"
    #include "pretty/pretty.h"
}

using namespace cl::sycl;
using namespace std;

#define GET_PLATFORM_INFO(plat, prop) plat.get_info<info::platform::prop>()
#define GET_DEVICE_INFO(dev, prop) dev.get_info<info::device::prop>()
#define GET_DEVICE_INFO_INT(dev, prop) to_string(dev.get_info<info::device::prop>())
#define BOOL_TO_YES_NO(x) ((x) ? "yes" : "no")

#define GET_DEVICE_KEY_VAL(dev, key, prop) \
    {key, GET_DEVICE_INFO(dev, prop)}
#define GET_DEVICE_KEY_VAL_BOOL(dev, key, prop) \
    {key, BOOL_TO_YES_NO(GET_DEVICE_INFO(dev, prop))}
#define GET_DEVICE_KEY_VAL_INT(dev, key, prop) \
    {key, to_string(GET_DEVICE_INFO(dev, prop))}

map<info::device_type, string>
device_type_to_string = {
    {info::device_type::accelerator, "Accelerator"},
    {info::device_type::cpu, "CPU"}
};

map<info::global_mem_cache_type, string>
global_mem_cache_type_to_string = {
    {info::global_mem_cache_type::none, "None"},
    {info::global_mem_cache_type::read_only, "Read-only"},
    {info::global_mem_cache_type::read_write, "Read-write"}
};

map<info::local_mem_type, string>
local_mem_type_to_string  = {
    {info::local_mem_type::none, "None"},
    {info::local_mem_type::local, "Local"},
    {info::local_mem_type::global, "Global"}
};

map<aspect, string> aspect_to_string = {
    {aspect::cpu, "cpu"},
    {aspect::fp16, "fp16"},
    {aspect::fp64, "fp64"},
    {aspect::accelerator, "accelerator"},
    {aspect::online_compiler, "online_compiler"},
    {aspect::online_linker, "online_linker"},
    {aspect::queue_profiling, "queue_profiling"},
    {aspect::usm_host_allocations, "usm_host_allocations"},
    {aspect::usm_shared_allocations, "usm_shared_allocations"}
};

static string
list_aspects(device dev) {
    string s = "";
    for (auto [key, val] : aspect_to_string) {
        if (dev.has(key)) {
            s += val + " ";
        }
    }
    return s;
}

int
main() {
    pretty_printer *pp = pp_init();
    pp->key_width = 30;
    auto platforms = platform::get_platforms();
    for (auto plat : platforms) {
        vector<array<string, 2>> plat_props = {
            {"Platform", GET_PLATFORM_INFO(plat, name)},
            {"Vendor", plat.get_info<info::platform::vendor>()},
            {"Version", plat.get_info<info::platform::version>()},
            {"Profile", plat.get_info<info::platform::profile>()},
        };
        for (auto p : plat_props) {
            pp_print_key_value(pp, p[0].c_str(), "%s", p[1].c_str());
        }
        printf("\n");
        pp->indent++;
        for (auto dev : plat.get_devices()) {
            auto type = GET_DEVICE_INFO(dev, device_type);
            auto gmem_cache = GET_DEVICE_INFO(dev, global_mem_cache_type);
            auto lmem_type = GET_DEVICE_INFO(dev, local_mem_type);
            auto aspects = list_aspects(dev);
            vector<array<string, 2>> dev_props = {
                GET_DEVICE_KEY_VAL(dev, "Device", name),
                GET_DEVICE_KEY_VAL_BOOL(dev, "Available", is_available),
                GET_DEVICE_KEY_VAL(dev, "Driver version", driver_version),
                GET_DEVICE_KEY_VAL(dev, "Version", version),
                GET_DEVICE_KEY_VAL(dev, "OpenCL C version", opencl_c_version),
                {"Type", device_type_to_string[type]},
                GET_DEVICE_KEY_VAL_INT(dev, "Vendor ID", vendor_id),
                GET_DEVICE_KEY_VAL_INT(dev, "Max compute units", max_compute_units),
                GET_DEVICE_KEY_VAL_INT(dev, "Address bits", address_bits),
                {"Global mem. cache", global_mem_cache_type_to_string[gmem_cache]},
                {"Local memory type", local_mem_type_to_string[lmem_type]},
                {"Aspects", aspects},
            };
            for (auto p : dev_props) {
                pp_print_key_value(pp, p[0].c_str(), "%s", p[1].c_str());
            }

            const char *keys[] = {
                "Local memory size",
                "Max clock freq.",
                "Max allocation",
                "Global mem cache size",
                "Global mem cache line size",
            };
            const char *suffixes[] = {"B", "Hz", "B", "B", ""};
            uint64_t values[] = {
                GET_DEVICE_INFO(dev, local_mem_size),
                GET_DEVICE_INFO(dev, max_clock_frequency) * 1000 * 1000,
                GET_DEVICE_INFO(dev, max_mem_alloc_size),
                GET_DEVICE_INFO(dev, global_mem_cache_size),
                GET_DEVICE_INFO(dev, global_mem_cache_line_size)
            };
            for (size_t i = 0; i < ARRAY_SIZE(keys); i++) {
                const char *suf = suffixes[i];
                const char *key = keys[i];
                pp->n_decimals = !strcmp(suf, "") ? 0 : 2;
                pp_print_key_value_with_unit(pp, key, (double)values[i], suf);
            }

            const char *types[] = {
                "char", "float", "double", "half"
            };
            uint64_t native_values[] = {
                GET_DEVICE_INFO(dev, native_vector_width_char),
                GET_DEVICE_INFO(dev, native_vector_width_float),
                GET_DEVICE_INFO(dev, native_vector_width_double),
                GET_DEVICE_INFO(dev, native_vector_width_half),
            };
            uint64_t preferred_values[] = {
                GET_DEVICE_INFO(dev, preferred_vector_width_char),
                GET_DEVICE_INFO(dev, preferred_vector_width_float),
                GET_DEVICE_INFO(dev, preferred_vector_width_double),
                GET_DEVICE_INFO(dev, native_vector_width_half)
            };
            pp->n_decimals = 0;
            pp_print_printf(pp, "Native vector widths\n");
            pp->indent++;
            for (size_t i = 0; i < ARRAY_SIZE(types); i++) {
                pp_print_key_value_with_unit(pp, types[i], (double)native_values[i], "");
            }
            pp->indent--;
            pp_print_printf(pp, "Preferred vector widths\n");
            pp->indent++;
            for (size_t i = 0; i < ARRAY_SIZE(types); i++) {
                pp_print_key_value_with_unit(pp, types[i], (double)preferred_values[i], "");
            }
            pp->indent--;
        }
        pp->indent--;
        printf("\n");
    }
    pp_free(pp);
    return 0;
}
