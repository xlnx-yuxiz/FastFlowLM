# Local source patch for xgrammar 97787376faee5ed8466cfad57c99855e4ce2f6aa.
# Keep the submodule untouched, including shared dependency caches. This overlay
# is used by both the application and standalone tests through chat_sources.cmake.
function(flm_patch_xgrammar_string_bounds original output)
    file(READ "${original}" upstream)
    # Git checkouts on Windows may use CRLF. Pin the normalized source contents.
    string(REPLACE "\r\n" "\n" upstream "${upstream}")
    string(SHA256 upstream_sha "${upstream}")
    if(NOT upstream_sha STREQUAL "f9d2ef9ca78c4adabb514580b4ce1f13af7ba6189d2faa4f5a1217c2727b8782")
        message(FATAL_ERROR "FLM xgrammar string-bounds patch: upstream source changed (${upstream_sha}); review the patch before updating xgrammar")
    endif()

    set(before [=[      int32_t character =
          builder_.AddCharacterClass({{'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}}, true);
]=])
    set(after [=[      // One atom is one decoded Unicode scalar, regardless of JSON spelling.
      // Do not count a surrogate pair as two characters or allow unpaired surrogates.
      int32_t raw = builder_.AddCharacterClass(
          {{0, 0x1f}, {'"', '"'}, {'\\', '\\'}, {0xd800, 0xdfff}}, true
      );
      int32_t hex = builder_.AddCharacterClass({{'0', '9'}, {'a', 'f'}, {'A', 'F'}});
      int32_t d = builder_.AddCharacterClass({{'d', 'd'}, {'D', 'D'}});
      int32_t bmp_prefix = Choice({
          Sequence({builder_.AddCharacterClass(
                        {{'0', '9'}, {'a', 'c'}, {'A', 'C'}, {'e', 'f'}, {'E', 'F'}}), hex}),
          Sequence({d, builder_.AddCharacterClass({{'0', '7'}})})
      });
      int32_t bmp = Sequence({ByteString("\\u"), bmp_prefix, hex, hex});
      int32_t pair = Sequence({
          ByteString("\\u"), d,
          builder_.AddCharacterClass({{'8', '9'}, {'a', 'b'}, {'A', 'B'}}), hex, hex,
          ByteString("\\u"), d,
          builder_.AddCharacterClass({{'c', 'f'}, {'C', 'F'}}), hex, hex
      });
      int32_t simple = Sequence({ByteString("\\"), builder_.AddCharacterClass({
          {'"', '"'}, {'\\', '\\'}, {'/', '/'}, {'b', 'b'}, {'f', 'f'},
          {'n', 'n'}, {'r', 'r'}, {'t', 't'}
      })});
      int32_t character = Choice({raw, simple, bmp, pair});
]=])
    string(FIND "${upstream}" "${before}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "FLM xgrammar string-bounds patch: expected source block missing")
    endif()
    string(REPLACE "${before}" "${after}" patched "${upstream}")
    set(existing "")
    if(EXISTS "${output}")
        file(READ "${output}" existing)
    endif()
    if(NOT existing STREQUAL patched)
        get_filename_component(output_dir "${output}" DIRECTORY)
        file(MAKE_DIRECTORY "${output_dir}")
        file(WRITE "${output}" "${patched}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${original}")
    string(SHA256 patched_sha "${patched}")
    message(STATUS "FLM xgrammar string-bounds patch: ${upstream_sha} -> ${patched_sha}")
endfunction()
