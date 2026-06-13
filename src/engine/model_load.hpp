#pragma once

#include "llama.h"

// MinGW builds may target a Windows version where PrefetchVirtualMemory is unavailable
// when mmap is enabled. Disabling mmap avoids hard load failures on Windows.
inline llama_model_params llm_trace_model_params() {
    llama_model_params params = llama_model_default_params();
#if defined(_WIN32)
    params.use_mmap = false;
#endif
    return params;
}
