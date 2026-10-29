cmake_minimum_required(VERSION 3.22)
project(model_backend_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(XRT_INCLUDE_DIR "/opt/xilinx/xrt/include" CACHE PATH "XRT headers")
set(XRT_LIB_DIR "${XRT_INCLUDE_DIR}/../lib" CACHE PATH "XRT libraries")
find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp)

add_executable(test_model_backend test_model_backend.cpp
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp")
target_include_directories(test_model_backend PRIVATE
    "${FLM_SOURCE_DIR}/include" "${XRT_INCLUDE_DIR}")
if(BOOST_INCLUDE_DIR)
    target_include_directories(test_model_backend PRIVATE "${BOOST_INCLUDE_DIR}")
endif()
target_link_directories(test_model_backend PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(test_model_backend PRIVATE xrt_coreutil)
if(WIN32)
    # Match the CLI's XRT header contract; this SDK ships without version-slim.h.
    target_compile_definitions(test_model_backend PRIVATE
        DISABLE_ABI_CHECK=1 WIN32_LEAN_AND_MEAN NOMINMAX __WINDOWS__)
endif()

enable_testing()
add_test(NAME backend_registry COMMAND test_model_backend)
foreach(mode enabled disabled)
    add_executable(test_backend_selection_${mode} test_backend_selection.cpp)
    target_include_directories(test_backend_selection_${mode} PRIVATE "${FLM_SOURCE_DIR}/include")
    if(mode STREQUAL "enabled")
        target_compile_definitions(test_backend_selection_${mode} PRIVATE FLM_ENABLE_HIP_OGA=1)
    endif()
    add_test(NAME backend_selection_${mode} COMMAND test_backend_selection_${mode})
endforeach()
