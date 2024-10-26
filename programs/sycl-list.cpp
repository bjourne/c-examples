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
#include "pretty/pretty.h"
}

using namespace cl::sycl;
using namespace std;

#define GET_DEVICE_INFO(dev, prop) dev.get_info<info::device::prop>()
#define GET_DEVICE_INFO_INT(dev, prop) to_string(dev.get_info<info::device::prop>())

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
    {aspect::accelerator, "accelerator"},
    {aspect::online_compiler, "online_compiler"},
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
    pp->key_width = 22;
    auto platforms = platform::get_platforms();
    for (auto plat : platforms) {
        vector<array<string, 2>> plat_props = {
            {"Platform", plat.get_info<info::platform::name>()},
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
            auto vend = GET_DEVICE_INFO(dev, vendor_id);
            auto type = GET_DEVICE_INFO(dev, device_type);
            auto max_compute = GET_DEVICE_INFO_INT(dev, max_compute_units);
            auto addr_bits = GET_DEVICE_INFO_INT(dev, address_bits);
            auto gmem_cache = GET_DEVICE_INFO(dev, global_mem_cache_type);
            auto lmem_type = GET_DEVICE_INFO(dev, local_mem_type);
            auto aspects = list_aspects(dev);
            vector<array<string, 2>> dev_props = {
                {"Device", dev.get_info<info::device::name>()},
                {"Driver version", dev.get_info<info::device::driver_version>()},
                {"Version", dev.get_info<info::device::version>()},
                {"OpenCL C version", dev.get_info<info::device::opencl_c_version>()},
                {"Type", device_type_to_string[type]},
                {"Vendor ID", to_string(vend)},
                {"Max compute units", max_compute},
                {"Address bits", addr_bits},
                {"Global mem. cache", global_mem_cache_type_to_string[gmem_cache]},
                {"Local memory type", local_mem_type_to_string[lmem_type]},
                {"Aspects", aspects},
            };
            for (auto p : dev_props) {
                pp_print_key_value(pp, p[0].c_str(), "%s", p[1].c_str());
            }
            auto lmem_size = GET_DEVICE_INFO(dev, local_mem_size);
            auto freq = GET_DEVICE_INFO(dev, max_clock_frequency);
            pp_print_key_value_with_unit(pp, "Local memory size", (double)lmem_size, "B");
            pp_print_key_value_with_unit(pp, "Max clock freq.", (double)freq * 1000 * 1000, "Hz");
        }
        pp->indent--;
        printf("\n");
    }
    pp_free(pp);
    return 0;
}
