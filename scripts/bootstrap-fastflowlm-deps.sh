#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

expected_tokenizers_sha="$(git rev-parse HEAD:third_party/tokenizers-cpp)"
tokenizers_url="$(git config --file .gitmodules --get submodule.third_party/tokenizers-cpp.url)"

if ! git submodule sync --recursive third_party/tokenizers-cpp ||
   ! git submodule update --init --recursive third_party/tokenizers-cpp; then
    printf 'git submodule update could not write parent git metadata; falling back to a direct clone\n' >&2
    if [[ -e third_party/tokenizers-cpp/.git ]]; then
        git -C third_party/tokenizers-cpp fetch --tags "$tokenizers_url" "$expected_tokenizers_sha"
    else
        if [[ -d third_party/tokenizers-cpp ]] &&
           [[ -n "$(find third_party/tokenizers-cpp -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            printf 'third_party/tokenizers-cpp is non-empty and is not a git checkout\n' >&2
            exit 1
        fi
        rmdir third_party/tokenizers-cpp 2>/dev/null || true
        git clone "$tokenizers_url" third_party/tokenizers-cpp
    fi
    git -C third_party/tokenizers-cpp checkout "$expected_tokenizers_sha"
    git -C third_party/tokenizers-cpp submodule update --init --recursive
fi

actual_tokenizers_sha="$(git -C third_party/tokenizers-cpp rev-parse HEAD)"
if [[ "$actual_tokenizers_sha" != "$expected_tokenizers_sha" ]]; then
    printf 'tokenizers-cpp submodule mismatch: expected %s, got %s\n' \
        "$expected_tokenizers_sha" "$actual_tokenizers_sha" >&2
    exit 1
fi

if [[ ! -f third_party/tokenizers-cpp/CMakeLists.txt ]]; then
    printf 'tokenizers-cpp submodule is initialized but CMakeLists.txt is missing\n' >&2
    exit 1
fi

printf 'tokenizers-cpp ready at %s\n' "$actual_tokenizers_sha"
