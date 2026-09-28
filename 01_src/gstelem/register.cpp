#include "gstelem/register.hpp"

#include <gst/gst.h>

#include <mutex>

#include "gstelem/sand_letterbox_element.hpp"

namespace yvo::gst {

bool register_elements() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        ok = register_sand_letterbox(nullptr);
        // stage 2b: ok = register_hailo_sink(nullptr) && ok;
    });
    return ok;
}

} // namespace yvo::gst
