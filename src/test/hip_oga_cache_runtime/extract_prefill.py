#!/usr/bin/env python3
"""Compile the production prefill transaction unchanged in a test-only adapter.

Extraction fails closed when source boundaries change; the generated receipt pins
the full source and fragment. No shipping source or runtime controls are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
p.add_argument('output',type=Path)
args=p.parse_args()
source=args.source.read_text(encoding='utf-8')
start_marker='        auto prefill_start_time = this->profiler_list[PREFILL_TIME].start();'
end_marker='    } else {\n        // HIP OGA media prefill transaction:'
assert source.count(end_marker)==1
end=source.index(end_marker)
start=source.rfind(start_marker,0,end)
assert start>=source.index('bool HipOgaModel::insert(')
fragment=source[start:end]
# Both the recovery path and the following accounting/commit belong to the test.
assert 'prefix_cache_.commit_checkpoint(' in fragment
assert 'meta_info.cached_prompt_tokens =' in fragment
assert fragment.count('catch (const std::exception& e)')==1
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'production_prefill.inc').write_text(fragment,encoding='utf-8')
receipt=dict(source=str(args.source.resolve()),
    sourceSha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
    fragmentSha256=hashlib.sha256(fragment.encode()).hexdigest(),
    firstLine=source[:start].count('\n')+1,lastLine=source[:end].count('\n'),
    scope='Exact production prefill recovery, accounting and checkpoint commit; test-owned OGA adapter')
(args.output/'extraction.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))

media_start_marker='        // BEGIN HIP OGA MEDIA TRANSACTION (compiled unchanged by its test adapter)\n'
media_end_marker='        // END HIP OGA MEDIA TRANSACTION'
assert source.count(media_start_marker)==source.count(media_end_marker)==1
media_start=source.index(media_start_marker)+len(media_start_marker)
media_end=source.index(media_end_marker,media_start)
media=source[media_start:media_end]
assert 'prefix_cache_.commit_checkpoint(' in media
assert 'std::move(prepared.identity)' in media
assert 'if (is_cancelled())' in media
assert media.count('hip_engine->begin_multimodal_turn(')==2
(args.output/'production_media_prefill.inc').write_text(media,encoding='utf-8')
(args.output/'media-extraction.json').write_text(json.dumps(dict(
    sourceSha256=receipt['sourceSha256'],fragmentSha256=hashlib.sha256(media.encode()).hexdigest(),
    firstLine=source[:media_start].count('\n')+1,lastLine=source[:media_end].count('\n'),
    scope='Exact production media restore/fallback/cancellation/accounting/commit'),indent=2))

token_start=source.index('std::vector<int> read_oga_token_tensor(')
token_end=source.index('\nflm::hip_oga::MediaTensorIdentity copy_oga_media_tensor(',token_start)
token_fragment=source[token_start:token_end]
(args.output/'production_media_tokens.inc').write_text(token_fragment,encoding='utf-8')
(args.output/'media-tokens-extraction.json').write_text(json.dumps(dict(
    sourceSha256=receipt['sourceSha256'],fragmentSha256=hashlib.sha256(token_fragment.encode()).hexdigest(),
    scope='Exact production OGA processor token tensor shape/type/range validation'),indent=2))

tensor_start_marker='flm::hip_oga::MediaTensorIdentity copy_oga_media_tensor('
tensor_end_marker='\nstruct HipOgaPreparedMedia {'
assert source.count(tensor_start_marker)==1
assert source.count(tensor_end_marker)==1
tensor_start=source.index(tensor_start_marker)
tensor_end=source.index(tensor_end_marker,tensor_start)
tensor_fragment=source[tensor_start:tensor_end]
assert 'identity.bytes.assign(data, data + bytes);' in tensor_fragment
assert 'inputs.Get(name)' in tensor_fragment
(args.output/'production_media_tensor.inc').write_text(tensor_fragment,encoding='utf-8')

identity_start_marker='        if (capture_identity && media_rewind_qualified_) {\n'
identity_end_marker='        return prepared;'
assert source.count(identity_start_marker)==1
identity_start=source.index(identity_start_marker)
identity_end=source.index(identity_end_marker,identity_start)
identity_fragment=source[identity_start:identity_end]
assert 'prepared.inputs->GetNames()' in identity_fragment
assert identity_fragment.count('copy_oga_media_tensor(*prepared.inputs,')==2
assert '"pixel_values"' in identity_fragment
assert 'std::move(prepared.loaded_images.buffers)' in identity_fragment
(args.output/'production_media_identity.inc').write_text(identity_fragment,encoding='utf-8')
prepare_start=source.index('        prepared.tokens = read_oga_token_tensor(*prepared.inputs, "input_ids");')
prepare_fragment=source[prepare_start:identity_end]
(args.output/'production_media_prepare.inc').write_text(prepare_fragment,encoding='utf-8')
(args.output/'media-identity-extraction.json').write_text(json.dumps(dict(
    sourceSha256=receipt['sourceSha256'],
    tensorCopyFragmentSha256=hashlib.sha256(tensor_fragment.encode()).hexdigest(),
    identityCaptureFragmentSha256=hashlib.sha256(identity_fragment.encode()).hexdigest(),
    preparedInputsFragmentSha256=hashlib.sha256(prepare_fragment.encode()).hexdigest(),
    tensorCopyFirstLine=source[:tensor_start].count('\n')+1,
    tensorCopyLastLine=source[:tensor_end].count('\n'),
    identityCaptureFirstLine=source[:identity_start].count('\n')+1,
    identityCaptureLastLine=source[:identity_end].count('\n'),
    scope='Exact production media tensor copy and processor identity capture'),indent=2))

# Exercise package metadata through the constructor's actual file reader and
# validation block; direct JSON-value tests cannot catch swallowed parse errors.
reader_start=source.index('nlohmann::json read_json_file_if_present(')
reader_end=source.index('\nflm::hip_oga::DecoderCanonicalizationPolicy',reader_start)
reader_fragment=source[reader_start:reader_end]
config_start=source.index('        const auto media_compatibility = ')
config_end=source.index('        config_directory_ = ',config_start)
config_fragment=source[config_start:config_end]
(args.output/'production_json_reader.inc').write_text(reader_fragment,encoding='utf-8')
(args.output/'production_media_config.inc').write_text(config_fragment,encoding='utf-8')
(args.output/'media-config-extraction.json').write_text(json.dumps(dict(
    sourceSha256=receipt['sourceSha256'],
    readerFragmentSha256=hashlib.sha256(reader_fragment.encode()).hexdigest(),
    configFragmentSha256=hashlib.sha256(config_fragment.encode()).hexdigest(),
    scope='Exact production package metadata file read, validation and signature'),indent=2))
