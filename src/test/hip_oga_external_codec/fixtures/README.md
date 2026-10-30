# HIP OGA External Tokenizer Fixtures

`text_cases.json` defines deterministic no-hardware prompts for the external
tokenizer parity lane. The cases are intentionally model-weight free: real OGA
and FastFlowLM token IDs are captured later by the integration or hardware
owner, while this lane validates the fixture contract and evidence format.

Required coverage:

- Qwen2.5 byte-level BPE family
- Gemma SentencePiece family
- Plain prompt, chat prompt, tool-call prompt, code, whitespace, CJK, emoji,
  and added-token literals

Every case either carries token IDs from an explicit capture or marks
`capture_required: true`. The lane must not infer token IDs from another model
family.
