#include "gstelem/register.hpp"

#include <gst/gst.h>

#include <mutex>

#include "gstelem/sand_letterbox_element.hpp"
#ifdef YVO_GSTELEM_HAVE_HAILO
#include "gstelem/hailo_infer_sink.hpp"
#endif

namespace yvo::gst {

bool register_elements() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        ok = register_sand_letterbox(nullptr);
#ifdef YVO_GSTELEM_HAVE_HAILO
        ok = register_hailo_sink(nullptr) && ok;
#endif
    });
    return ok;
}

} // namespace yvo::gst
