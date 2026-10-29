cmake_minimum_required(VERSION 3.21)
project(hip_oga_cache_runtime LANGUAGES CXX)
if(NOT FLM_SOURCE_ROOT)
  get_filename_component(FLM_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
file(TO_CMAKE_PATH "${FLM_SOURCE_ROOT}" FLM_SOURCE_ROOT)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${FLM_SOURCE_ROOT}/src/common/AutoModel/hip_oga_model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/extract_prefill.py")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract_prefill.py"
  "${FLM_SOURCE_ROOT}/src/common/AutoModel/hip_oga_model.cpp" "${CMAKE_CURRENT_BINARY_DIR}"
  RESULT_VARIABLE extraction_result)
if(NOT extraction_result EQUAL 0)
  message(FATAL_ERROR "Production prefill extraction failed")
endif()
add_executable(test_cache_runtime test_cache_runtime.cpp)
target_compile_features(test_cache_runtime PRIVATE cxx_std_17)
target_include_directories(test_cache_runtime PRIVATE "${FLM_SOURCE_ROOT}/src/include"
  "${ONNXRUNTIME_GENAI_HOME}/include" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(test_cache_runtime PRIVATE "${ONNXRUNTIME_GENAI_HOME}/lib/onnxruntime-genai.lib")

add_executable(test_hip_oga_prefix_cache
  "${FLM_SOURCE_ROOT}/src/test/hip_oga_backend_static/test_hip_oga_prefix_cache.cpp")
target_compile_features(test_hip_oga_prefix_cache PRIVATE cxx_std_17)
target_include_directories(test_hip_oga_prefix_cache PRIVATE "${FLM_SOURCE_ROOT}/src/include")
enable_testing()
add_test(NAME hip_oga_prefix_cache_policy COMMAND test_hip_oga_prefix_cache)

add_executable(test_hip_oga_media_prefix_cache
  "${FLM_SOURCE_ROOT}/src/test/hip_oga_backend_static/test_hip_oga_media_prefix_cache.cpp")
target_compile_features(test_hip_oga_media_prefix_cache PRIVATE cxx_std_17)
target_include_directories(test_hip_oga_media_prefix_cache PRIVATE "${FLM_SOURCE_ROOT}/src/include")
add_test(NAME hip_oga_media_prefix_cache_policy COMMAND test_hip_oga_media_prefix_cache)

add_executable(test_media_transaction test_media_transaction.cpp)
target_compile_features(test_media_transaction PRIVATE cxx_std_17)
target_include_directories(test_media_transaction PRIVATE "${FLM_SOURCE_ROOT}/src/include"
  "${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME hip_oga_media_transaction COMMAND test_media_transaction)

add_executable(test_hip_oga_media_inputs
  "${FLM_SOURCE_ROOT}/src/test/hip_oga_backend_static/test_hip_oga_media_inputs.cpp")
target_compile_features(test_hip_oga_media_inputs PRIVATE cxx_std_17)
target_include_directories(test_hip_oga_media_inputs PRIVATE "${FLM_SOURCE_ROOT}/src/include")
add_test(NAME hip_oga_media_inputs COMMAND test_hip_oga_media_inputs)

foreach(policy_test IN ITEMS session_policy config_directory)
  add_executable(test_hip_oga_${policy_test}
    "${FLM_SOURCE_ROOT}/src/test/hip_oga_backend_static/test_hip_oga_${policy_test}.cpp")
  target_compile_features(test_hip_oga_${policy_test} PRIVATE cxx_std_17)
  target_include_directories(test_hip_oga_${policy_test} PRIVATE "${FLM_SOURCE_ROOT}/src/include")
  add_test(NAME hip_oga_${policy_test} COMMAND test_hip_oga_${policy_test})
endforeach()
