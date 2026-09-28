// Static registration of the research GStreamer elements (no plugin .so): call after gst_init(), before
// gst_parse_launch / gst_element_factory_make. Idempotent and thread-safe.
//   yvosandletterbox  — stage 2a (sand_letterbox_element.hpp)
//   yvohailosink      — stage 2b (to be added here)
#pragma once

namespace yvo::gst {

// Returns false if any element failed to register (the failure is logged by GStreamer).
bool register_elements();

} // namespace yvo::gst
