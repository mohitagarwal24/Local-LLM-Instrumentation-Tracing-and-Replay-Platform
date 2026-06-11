cmake_minimum_required(VERSION 3.20)
project(llm-trace-platform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/deps.cmake)

set(LLM_TRACE_SOURCES
    src/main.cpp
    src/tracer/tracer.cpp
    src/tracer/topology.cpp
    src/tracer/recorder.cpp
    src/engine/llama_runner.cpp
    src/tui/app.cpp
    src/tui/panels/topology_panel.cpp
    src/tui/panels/stream_panel.cpp
    src/tui/panels/attention_panel.cpp
    src/tui/panels/metrics_panel.cpp
    src/tui/panels/anomaly_panel.cpp
)

add_executable(llm-trace ${LLM_TRACE_SOURCES})

target_include_directories(llm-trace PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(llm-trace PRIVATE
    llama
    ftxui::screen
    ftxui::dom
    ftxui::component
)

if(MINGW)
    target_link_libraries(llm-trace PRIVATE ws2_32)
endif()

add_executable(generate-demo-trace src/tools/generate_demo_trace.cpp src/tracer/recorder.cpp)
target_include_directories(generate-demo-trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
