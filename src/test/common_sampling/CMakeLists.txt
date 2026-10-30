cmake_minimum_required(VERSION 3.22)
project(common_sampling_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(XRT_INCLUDE_DIR "" CACHE PATH "XRT include directory used by buffer.hpp")
set(XRT_LIB_DIR "" CACHE PATH "XRT library directory")
set(BOOST_INCLUDE_DIR "" CACHE PATH "Boost include directory")
option(FLM_TEST_NATIVE_SAMPLER "Also build the native buffer/Sampler compatibility test" OFF)

add_executable(test_common_sampling test_common_sampling.cpp)
target_include_directories(test_common_sampling PRIVATE "${FLM_SOURCE_DIR}/include")
target_compile_features(test_common_sampling PRIVATE cxx_std_17)
enable_testing()
add_test(NAME common_sampling COMMAND test_common_sampling)

foreach(target IN ITEMS test_sampling_request test_oga_common_mapping)
add_executable(${target} ${target}.cpp)
target_include_directories(${target} PRIVATE "${FLM_SOURCE_DIR}/include")
target_compile_features(${target} PRIVATE cxx_std_17)
add_test(NAME ${target} COMMAND ${target})
endforeach()

if(FLM_TEST_NATIVE_SAMPLER)
if(NOT IS_DIRECTORY "${XRT_INCLUDE_DIR}")
    message(FATAL_ERROR "Set -DXRT_INCLUDE_DIR to the installed XRT include directory")
endif()
if(NOT IS_DIRECTORY "${XRT_LIB_DIR}")
    message(FATAL_ERROR "Set -DXRT_LIB_DIR to the installed XRT library directory")
endif()
if(NOT IS_DIRECTORY "${BOOST_INCLUDE_DIR}")
    find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp REQUIRED)
endif()
foreach(target IN ITEMS test_native_compat test_native_common)
add_executable(${target} ${target}.cpp
    "${FLM_SOURCE_DIR}/common/modules/sampler.cpp")
target_include_directories(${target} PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${XRT_INCLUDE_DIR}"
    "${BOOST_INCLUDE_DIR}")
target_compile_features(${target} PRIVATE cxx_std_17)
target_compile_definitions(${target} PRIVATE USEAVX2=0)
target_link_directories(${target} PRIVATE "${XRT_LIB_DIR}")

find_library(XRT_COREUTIL_LIBRARY NAMES xrt_coreutil PATHS "${XRT_LIB_DIR}" NO_DEFAULT_PATH)
if(XRT_COREUTIL_LIBRARY)
    target_link_libraries(${target} PRIVATE "${XRT_COREUTIL_LIBRARY}")
endif()

if(MSVC)
    target_compile_options(${target} PRIVATE /arch:AVX2)
    target_compile_definitions(${target} PRIVATE
        DISABLE_ABI_CHECK=1 WIN32_LEAN_AND_MEAN NOMINMAX __WINDOWS__)
endif()

add_test(NAME ${target} COMMAND ${target})
endforeach()
endif()
