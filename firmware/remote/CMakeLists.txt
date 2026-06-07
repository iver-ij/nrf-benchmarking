cmake_minimum_required(VERSION 3.20.0)
# Work around legacy CMakeLists in transitive OpenAMP/libmetal modules.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tinyml_dsp_remote)

target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_sources(app PRIVATE
    src/main.c
    ../src/mel_constants.c
)
