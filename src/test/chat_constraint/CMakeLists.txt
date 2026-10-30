cmake_minimum_required(VERSION 3.20)
project(flm_chat_constraint_tests LANGUAGES CXX)
enable_testing()
get_filename_component(FLM_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${FLM_SOURCE}/common/chat/chat_sources.cmake")
add_executable(test_chat_constraint test_chat_constraint.cpp test_chat_string_bounds.cpp
    "${FLM_SOURCE}/common/chat/chat_constraint.cpp"
    "${FLM_SOURCE}/common/chat/chat_grammar.cpp"
    "${FLM_SOURCE}/common/chat/chat_format.cpp")
target_include_directories(test_chat_constraint PRIVATE "${FLM_SOURCE}/include")
target_link_libraries(test_chat_constraint PRIVATE flm_xgrammar)
target_compile_features(test_chat_constraint PRIVATE cxx_std_17)
add_executable(test_hip_oga_constraint_mask
    "${FLM_SOURCE}/test/hip_oga_backend_static/test_hip_oga_constraint_mask.cpp")
target_include_directories(test_hip_oga_constraint_mask PRIVATE "${FLM_SOURCE}/include")
target_compile_features(test_hip_oga_constraint_mask PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_chat_constraint PRIVATE /utf-8)
endif()
add_test(NAME constraint_mask COMMAND test_hip_oga_constraint_mask)
add_executable(test_chat_tool_history test_chat_tool_history.cpp)
target_include_directories(test_chat_tool_history PRIVATE "${FLM_SOURCE}/include")
target_compile_features(test_chat_tool_history PRIVATE cxx_std_17)
add_test(NAME chat_tool_history COMMAND test_chat_tool_history)
add_test(NAME xgrammar_source_patch COMMAND "${CMAKE_COMMAND}"
    "-DFLM_PATCH_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/patch-contract"
    -P "${CMAKE_CURRENT_LIST_DIR}/test_xgrammar_source_patch.cmake")
set_tests_properties(xgrammar_source_patch PROPERTIES FIXTURES_SETUP xgrammar_source_patch)
if(FLM_TEST_TOKENIZER_JSON)
    add_test(NAME chat_constraint COMMAND test_chat_constraint "${FLM_TEST_TOKENIZER_JSON}")
    set_tests_properties(chat_constraint PROPERTIES FIXTURES_REQUIRED xgrammar_source_patch)
endif()

add_executable(test_cli_tools "${FLM_SOURCE}/test/chat_output/test_cli_tools.cpp"
    "${FLM_SOURCE}/common/chat/cli_tools.cpp"
    "${FLM_SOURCE}/common/chat/chat_constraint.cpp"
    "${FLM_SOURCE}/common/chat/chat_grammar.cpp"
    "${FLM_SOURCE}/common/chat/chat_format.cpp")
target_include_directories(test_cli_tools PRIVATE "${FLM_SOURCE}/include")
target_link_libraries(test_cli_tools PRIVATE flm_xgrammar)
target_compile_features(test_cli_tools PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_cli_tools PRIVATE /utf-8)
endif()
add_test(NAME cli_tools COMMAND test_cli_tools)
