cmake_minimum_required(VERSION 3.20)
project(flm_hip_catalog_tests LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(FLM_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(CURL REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(test_options INTERFACE)
target_include_directories(test_options INTERFACE
    "${FLM_SOURCE}/include" "${FLM_SOURCE}/pull")
target_compile_definitions(test_options INTERFACE
    FLM_SOURCE_DIR="${FLM_SOURCE}"
    CMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}"
    CMAKE_XCLBIN_PREFIX="${FLM_SOURCE}/xclbins"
    __FLM_VERSION__="test" __NPU_VERSION__="0.0.0.0")
if(MSVC)
    target_compile_options(test_options INTERFACE /utf-8)
    target_compile_definitions(test_options INTERFACE
        __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1 _ENABLE_EXTENDED_ALIGNED_STORAGE
        NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

foreach(name model_list provider_overlay codec_options text_codec prefix_cache)
    add_executable(test_${name} test_hip_oga_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE test_options)
    add_test(NAME ${name} COMMAND test_${name})
endforeach()

add_executable(test_public_pull test_hip_oga_public_pull.cpp
    "${FLM_SOURCE}/pull/hip_oga_public_downloader.cpp"
    "${FLM_SOURCE}/pull/download_model.cpp")
target_link_libraries(test_public_pull PRIVATE test_options CURL::libcurl)
add_test(NAME public_pull COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/test_hip_oga_public_pull.py"
    --public-binary $<TARGET_FILE:test_public_pull>
    --overlay-binary $<TARGET_FILE:test_provider_overlay>)

# Exercise the existing upstream downloader contract through the same implementation.
add_executable(test_native_downloader "${FLM_SOURCE}/test/phi4_core/test_model_downloader.cpp"
    "${FLM_SOURCE}/pull/download_model.cpp"
    "${FLM_SOURCE}/pull/model_downloader.cpp"
    "${FLM_SOURCE}/common/utils.cpp")
target_include_directories(test_native_downloader PRIVATE
    "${FLM_SOURCE}/test/phi4_core" "${XRT_INCLUDE_DIR}" "${BOOST_INCLUDE_DIR}")
target_link_libraries(test_native_downloader PRIVATE test_options CURL::libcurl)
add_test(NAME native_downloader COMMAND test_native_downloader)
set_tests_properties(native_downloader PROPERTIES
    ENVIRONMENT "FLM_XCLBIN_PATH=${FLM_SOURCE}/xclbins")
add_test(NAME release_catalog COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/test_release_backend_catalog.py")

add_executable(test_download_transport test_download_transport.cpp
    "${FLM_SOURCE}/pull/download_model.cpp")
target_link_libraries(test_download_transport PRIVATE test_options CURL::libcurl)
add_test(NAME download_transport COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/test_download_transport.py"
    --binary $<TARGET_FILE:test_download_transport>)
