#!/usr/bin/env python3
"""Offline checks for the sampling fixture's real endpoint response framing."""
import unittest

from test_hip_oga_sampling_api import parse_success_response
from check_sampling_matrix import cache_content_failures, diagnostics_follow_requests, readback_failures


class SamplingResponseParserTests(unittest.TestCase):
    def test_parses_chat_completion_json_and_sse(self):
        body = '{"object":"chat.completion","choices":[{"message":{"content":"red"}}],"usage":{"prompt_tokens_details":{"cached_tokens":2}}}'
        stream = 'data: {"choices":[{"delta":{"content":"red"}}]}\n\ndata: [DONE]\n\n'
        self.assertEqual(parse_success_response('/v1/chat/completions', False, body), ('red', 2))
        self.assertEqual(parse_success_response('/v1/chat/completions', True, stream), ('red', 0))

    def test_parses_completion_json_and_sse(self):
        body = '{"object":"text_completion","choices":[{"text":"blue"}],"usage":{"prompt_tokens_details":{"cached_tokens":0}}}'
        stream = 'data: {"choices":[{"text":"blue"}]}\n\ndata: [DONE]\n\n'
        self.assertEqual(parse_success_response('/v1/completions', False, body), ('blue', 0))
        self.assertEqual(parse_success_response('/v1/completions', True, stream), ('blue', 0))

    def test_parses_ollama_chat_json_and_ndjson(self):
        body = '{"message":{"role":"assistant","content":"green"},"done":true,"prompt_eval_count":4}'
        stream = '{"message":{"content":"green"},"done":false}\n{"message":{"content":""},"done":true}\n'
        self.assertEqual(parse_success_response('/api/chat', False, body), ('green', 0))
        self.assertEqual(parse_success_response('/api/chat', True, stream), ('green', 0))

    def test_parses_ollama_generate_json_and_ndjson(self):
        body = '{"response":"orange","done":true,"prompt_eval_count":4}'
        stream = '{"response":"ora","done":false}\n{"response":"nge","done":true}\n'
        self.assertEqual(parse_success_response('/api/generate', False, body), ('orange', 0))
        self.assertEqual(parse_success_response('/api/generate', True, stream), ('orange', 0))

    def test_rejects_unfinished_or_empty_streams_and_error_payloads(self):
        for endpoint, stream, raw in (
            ('/v1/chat/completions', True, 'data: {"choices":[{"delta":{"content":"x"}}]}\n'),
            ('/v1/completions', True, 'data: [DONE]\n'),
            ('/api/chat', True, '{"message":{"content":"x"},"done":false}\n'),
            ('/v1/chat/completions', False, '{"choices":[{"message":{"content":""}}]}'),
            ('/v1/completions', False, '{"choices":[{"text":""}]}'),
            ('/api/chat', True, '{"message":{"content":""},"done":true}\n'),
            ('/api/generate', True, '{"response":"x","done":false}\n'),
            ('/api/generate', False, '{"response":"x"}'),
            ('/api/generate', False, '{"error":"bad request","done":true}'),
            ('/api/generate', False, '{"response":"","done":true}'),
            ('/api/generate', False, '{"response":12,"done":true}'),
            ('/api/generate', True, '{"response":12,"done":true}\n'),
            ('/api/chat', False, '{"error":"bad request"}'),
            ('/v1/chat/completions', False, '{"error":{"message":"bad request"}}'),
        ):
            with self.subTest(endpoint=endpoint, stream=stream, raw=raw):
                with self.assertRaises(ValueError):
                    parse_success_response(endpoint, stream, raw)

    def test_matches_effective_and_runtime_diagnostics_to_request_order(self):
        records = [
            {'expected_settings': {'do_sample': True}, 'expected_runtime_settings': None},
            {'expected_settings': {'min_p': 0.2}, 'expected_runtime_settings': {'top_p': 1.0}},
        ]
        self.assertTrue(diagnostics_follow_requests(records, [('effective', {}), ('effective', {}), ('runtime', {})]))
        self.assertTrue(diagnostics_follow_requests(records, [('effective', {}), ('runtime', {}), ('effective', {})]))

    def test_rejects_extra_diagnostics_in_a_complete_log_stream(self):
        records = [{'expected_settings': {'do_sample': True}, 'expected_runtime_settings': None}]
        self.assertFalse(diagnostics_follow_requests(records, [('effective', {}), ('runtime', {})]))

    def test_rejects_wrong_neutral_runtime_settings(self):
        records = [{'name': 'min_p_only', 'expected_settings': {'min_p': 0.25},
                    'expected_runtime_settings': {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0,
                                                  'repetition_penalty': 1.0, 'do_sample': True,
                                                  'random_seed': 42}}]
        failures = readback_failures(records, [{'min_p': 0.25}],
                                     [{'temperature': 1.0, 'top_k': 20, 'top_p': 1.0,
                                       'repetition_penalty': 1.0, 'do_sample': True, 'random_seed': 42}])
        self.assertTrue(any('runtime top_k=20, expected 0' in failure for failure in failures))

    def test_rejects_missing_runtime_readback(self):
        records = [{'name': 'min_p_only', 'expected_settings': {'min_p': 0.25},
                    'expected_runtime_settings': {'temperature': 1.0}}]
        failures = readback_failures(records, [{'min_p': 0.25}], [])
        self.assertTrue(any('0 OGA runtime readbacks, expected 1' in failure for failure in failures))

    def test_rejects_wrong_effective_sampling_field(self):
        records = [{'name': 'min_p_only', 'expected_settings': {'min_p': 0.25},
                    'expected_runtime_settings': {'temperature': 1.0}}]
        failures = readback_failures(records, [{'min_p': 0.2}], [{'temperature': 1.0}])
        self.assertTrue(any('OGA min_p=0.2, expected 0.25' in failure for failure in failures))

    def test_cache_parity_skips_unseeded_and_one_sided_cases_but_checks_seeded_and_greedy(self):
        left = [
            {'name': 'min_p_only', 'content': 'red'},
            {'name': 'positive_frequency_penalty', 'content': 'one'},
            {'name': 'json_schema_color_enum', 'content': '{"color":"red"}'},
            {'name': 'greedy_follow', 'content': 'remember quartz'},
            {'name': 'settings_a', 'content': 'a green color'},
            {'name': 'extended_seeded_stream', 'content': 'blue'},
        ]
        right = [
            {'name': 'min_p_only', 'content': 'blue'},
            {'name': 'positive_frequency_penalty', 'content': 'two'},
            {'name': 'json_schema_color_enum', 'content': '{"color":"blue"}'},
            {'name': 'greedy_follow', 'content': 'remember quartz again'},
            {'name': 'settings_a', 'content': 'another green color'},
            {'name': 'extended_seeded_stream', 'content': 'green'},
            {'name': 'min_p_endpoint_api_chat_json', 'content': 'only in this lane'},
        ]
        failures = cache_content_failures(left, right, 'gemma/oga')
        self.assertEqual({failure.split('/')[2].split(':')[0] for failure in failures},
                         {'greedy_follow', 'settings_a', 'extended_seeded_stream'})


if __name__ == '__main__':
    unittest.main()
