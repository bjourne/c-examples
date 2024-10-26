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

#define BOOL_TO_YES_NO(x) ((x) ? "yes" : "no")

#define GET_PLATFORM_INFO(plat, prop) plat.get_info<info::platform::prop>()
#define GET_DEVICE_INFO(dev, prop) dev.get_info<info::device::prop>()

#define GET_DEVICE_INFO_BOOL(dev, prop) BOOL_TO_YES_NO(GET_DEVICE_INFO(dev, prop))
#define GET_DEVICE_INFO_INT(dev, prop) to_string(GET_DEVICE_INFO(dev, prop))
#define GET_DEVICE_INFO_STRINGS(dev, prop) join_strings(GET_DEVICE_INFO(dev, prop))
#define GET_DEVICE_INFO_MAPPED(dev, prop, map) map[GET_DEVICE_INFO(dev, prop)]

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
    {aspect::atomic64, "atomic64"},
    {aspect::accelerator, "accelerator"},
    {aspect::online_compiler, "online_compiler"},
    {aspect::online_linker, "online_linker"},
    {aspect::queue_profiling, "queue_profiling"},
    {aspect::usm_device_allocations, "usm_device_allocations"},
    {aspect::usm_host_allocations, "usm_host_allocations"},
    {aspect::usm_shared_allocations, "usm_shared_allocations"},
    {aspect::usm_system_allocations, "usm_system_allocations"}
};

map<info::fp_config, string> fp_config_to_string = {
    {info::fp_config::denorm, "denorm"},
    {info::fp_config::inf_nan, "inf_nan"},
    {info::fp_config::round_to_nearest, "round_to_nearest"},
    {info::fp_config::round_to_zero, "round_to_zero"},
    {info::fp_config::round_to_inf, "round_to_inf"},
    {info::fp_config::fma, "fma"},
    {info::fp_config::correctly_rounded_divide_sqrt, "correctly_rounded_divide_sqrt"},
    {info::fp_config::soft_float, "soft_float"}
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

static string
list_fp_config(vector<info::fp_config> config) {
    string join;
    for (auto e : config) {
        join += fp_config_to_string[e] + " ";
    }
    return join;
}

static string
join_strings(vector<string> l) {
    string s;
    for (auto e : l) {
        s += e + " ";
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
            auto aspects = list_aspects(dev);
            vector<array<string, 2>> dev_props = {
                {
                    "Device",
                    GET_DEVICE_INFO(dev, name)
                },
                {
                    "Available",
                    GET_DEVICE_INFO_BOOL(dev, is_available)
                },
                {
                    "Little endian",
                    GET_DEVICE_INFO_BOOL(dev, is_endian_little)
                },
                {
                    "Driver version",
                    GET_DEVICE_INFO(dev, driver_version)
                },
                {
                    "Version",
                    GET_DEVICE_INFO(dev, version)
                },
                {
                    "OpenCL C version",
                    GET_DEVICE_INFO(dev, opencl_c_version)
                },
                {
                    "Type",
                    GET_DEVICE_INFO_MAPPED(
                        dev,
                        device_type,
                        device_type_to_string
                    )
                },
                {
                    "Vendor ID",
                    GET_DEVICE_INFO_INT(dev, vendor_id)
                },
                {
                    "Max compute units",
                    GET_DEVICE_INFO_INT(dev, max_compute_units)
                },
                {
                    "Address bits",
                    GET_DEVICE_INFO_INT(dev, address_bits)
                },
                {
                    "Global mem. cache",
                    GET_DEVICE_INFO_MAPPED(
                        dev,
                        global_mem_cache_type,
                        global_mem_cache_type_to_string
                    )
                },
                {
                    "Local memory type",
                    GET_DEVICE_INFO_MAPPED(
                        dev,
                        local_mem_type,
                        local_mem_type_to_string
                    )

                },
                {"Aspects", aspects},
                {
                    "Extensions",
                    GET_DEVICE_INFO_STRINGS(dev, extensions)
                }
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
                "char", "short", "int", "long", "float", "double", "half"
            };
            uint64_t native_values[] = {
                GET_DEVICE_INFO(dev, native_vector_width_char),
                GET_DEVICE_INFO(dev, native_vector_width_short),
                GET_DEVICE_INFO(dev, native_vector_width_int),
                GET_DEVICE_INFO(dev, native_vector_width_long),
                GET_DEVICE_INFO(dev, native_vector_width_float),
                GET_DEVICE_INFO(dev, native_vector_width_double),
                GET_DEVICE_INFO(dev, native_vector_width_half),
            };
            uint64_t preferred_values[] = {
                GET_DEVICE_INFO(dev, preferred_vector_width_char),
                GET_DEVICE_INFO(dev, preferred_vector_width_short),
                GET_DEVICE_INFO(dev, preferred_vector_width_int),
                GET_DEVICE_INFO(dev, preferred_vector_width_long),
                GET_DEVICE_INFO(dev, preferred_vector_width_float),
                GET_DEVICE_INFO(dev, preferred_vector_width_double),
                GET_DEVICE_INFO(dev, preferred_vector_width_half)
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
            pp_print_printf(pp, "Floating point configurations\n");
            pp->indent++;

            string half_config = list_fp_config(GET_DEVICE_INFO(dev, half_fp_config));
            string single_config = list_fp_config(GET_DEVICE_INFO(dev, single_fp_config));
            string double_config = list_fp_config(GET_DEVICE_INFO(dev, double_fp_config));
            pp_print_key_value(pp, "half", half_config.c_str());
            pp_print_key_value(pp, "single", single_config.c_str());
            pp_print_key_value(pp, "double", double_config.c_str());
            pp->indent--;
        }
        pp->indent--;
        printf("\n");
    }
    pp_free(pp);
    return 0;
}
