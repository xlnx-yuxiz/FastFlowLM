# HIP OGA codec continuation

Implement the user-approved continuation on `feature/hip-oga-external-codec-demo`,
based on `demo/hip-oga-prefix-cache-experimental` (`2eb85e7b`).

- Add independent `oga`, `flm`, and `compare` codec selection for normal CLI
  `run` and `serve`. Keep legacy `external` as checked FLM output.
- Resolve per-layer CLI > combined CLI > environment > default OGA. Explicit
  CLI selection must override the legacy global compare environment flag.
- Instantiate/call native OGA codecs only when required by the selected mode;
  retain OGA generation and FastFlowLM/minja templates in every mode. Media uses
  the OGA processor and its decoder stream without external comparison.
- Keep codec metadata diagnostic only. Prefix reuse depends on actual token
  prefixes, backend context/capacity, and successful generator restoration.
- Verify configuration policy, existing codec fixtures, cache policy, and static
  integration guards. Build the normal Windows CLI in a separate stage, then
  run bounded hardware smoke checks if the existing host is available.
- Update documentation with current behavior and separate historical receipts
  from new-source verification. Review, commit, and push to the user's GitHub
  fork without modifying the demo branch.

## Verification outcome

- Configuration/legacy precedence, routing/codec fixtures, prefix policy and
  capacity, tokenizer probe, and whitespace checks passed.
- Fresh Windows normal CLI build passed. Both models passed all three modes;
  both FLM-only runs skipped native OGA tokenizer creation and comparison.
- Qwen FLM server two-turn check passed with 0 then 38 cached prompt tokens.
- Focused runtime/CLI review found no blocker; legacy mode help was clarified.
- Full receipt: `docs/provenance/hip-oga-codec-modes-20261023.json`.
