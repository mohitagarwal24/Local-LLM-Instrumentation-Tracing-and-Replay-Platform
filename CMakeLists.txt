cmake_minimum_required(VERSION 3.20)
project(llm-trace-platform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/deps.cmake)

add_executable(llm-trace
    src/main.cpp
)

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
