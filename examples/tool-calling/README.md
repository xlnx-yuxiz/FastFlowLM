# Native CLI tool calling

The development HIP/OGA CLI accepts an OpenAI function-definition array
directly. The default `--tool-choice required` preserves the existing demo flow
shown first; add `--tool-choice auto` explicitly to opt into structured auto,
validated on a pinned Qwen2.5 Coder 14B HIP package.

## Required mode (default)

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools calculator.json --tool-executor demo --prompt "Use add_numbers to add 17 and 28, then report the result."
```

`$Model` is a local OGA model directory or an installed HIP catalog tag. Start in this examples directory, or supply the full path to the JSON file. Use the project's HIP runtime setup before launching FLM.

Expected event shape (the final wording comes from the model):

```text
[call] add_numbers {"a":17,"b":28}
[result] {"result":45}
[call] finish_task {"answer":"The sum is 45."}
[answer] The sum is 45.
```

Omit `--prompt` for an interactive console. Each entered task starts a fresh tool conversation; the internal rounds retain the full tool history. `/load` changes the model while keeping the tool definitions and demo executor; `/bye` exits.

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools calculator.json --tool-executor demo
```

## Structured auto (opt-in)

Pass `--tool-choice auto` with `--tools`. The model can answer directly or
select one declared function for that response. With `--tool-executor demo`, FLM
runs the registered demo functions and asks the model what to do next; each
response selects at most one tool. This mode does not add `finish_task`.

For a greeting that needs no tool, structured auto can answer directly:

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools calculator.json --tool-choice auto --prompt "Say hello briefly."
```

To let the built-in demo executor perform an addition:

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools calculator.json --tool-choice auto --tool-executor demo --prompt "Use add_numbers to add 17 and 28, then report the result."
```

For the records example, the demo can look up a path, read that record, and add its values over sequential tool turns:

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools records.json --tool-choice auto --tool-executor demo --prompt "Look up the record named 'oak', read its path, add its a and b values with add_numbers, then report the label and sum."
```

Without `--tool-executor demo`, the CLI prints the assistant response as JSON,
including any generated tool call; it does not execute calls. A custom JSON
function definition only describes the function interface. To execute a custom
function, implement an API client or extend the native executor code; the CLI
currently exposes only the built-in `demo` executor listed below.

## Bringing your own definitions

`--tools FILE` loads a UTF-8 JSON array. Each item has `type: "function"`, a function `name`, optional `description`, and a JSON Schema `parameters` object. Names contain 1–64 ASCII letters, digits, underscores or hyphens. See the complete [calculator.json](calculator.json) and [weather.json](weather.json) examples.

With definitions alone, FLM selects one tool and prints an assistant JSON object containing its name and arguments:

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools weather.json --prompt "Call get_weather with city set exactly to Shanghai."
```

A definition describes an interface. Executing it requires a matching implementation. The explicit `--tool-executor demo` mode provides these built-in C++ implementations:

| Name | Required arguments | Execution |
| --- | --- | --- |
| `add_numbers` | `a`, `b`: integers | Add the supplied integers |
| `lookup_record` | `name`: string | Find a relative path in `data/index.json` |
| `read_record` | `path`: string | Read a JSON object under the data directory |

The demo executor requires the example signatures exactly; descriptions can be
changed and a subset can be selected. In required mode it adds
`finish_task(answer)` so the model can deliver its final answer as a tool call.
Structured auto returns the final answer as an ordinary message and does not add
`finish_task`. An arbitrary new function name has no built-in executor. For
external systems, use the API and implement execution in your client.

```powershell
flm run $Model --backend hip --ctx-len 2048 --tools records.json --tool-executor demo --prompt "Look up the record named 'oak', read its path, add its a and b values with add_numbers, then report the label and sum."
```

`--tool-data-dir DIR` overrides `data` beside the tool file. The supplied oak record yields `12 + (-5) = 7`.

## Current limits

- HIP text models; one tool call per response. Structured auto is limited to
  non-forced Hermes chat formats. The pinned Qwen2.5 Coder 14B package passed
  OGA/FLM/compare × cache off/on checks; other models require their own validation. The existing
  template-auto path remains model/template dependent; the pinned Qwen2.5 Coder
  package can emit function descriptions as ordinary text on a first turn.
  Parallel tool calls are outside this CLI flow.
- Definitions alone print a call in required mode, or a call/ordinary answer in auto mode. The demo mode also shows actual execution results and the final answer.
- Default 8 tool rounds, maximum 12 via `--max-tool-turns`; default 256 output tokens per round via `--max-tokens`. Explicit token limits are preserved.
- Tool file at most 64 KiB; serialized record result at most 4096 UTF-8 bytes. Record paths stay inside the configured data directory. Arithmetic inputs are integers within ±(2^53−1).
- One-shot errors return a nonzero exit status. Interactive errors return to the prompt.
- This is a native CLI path: it does not launch an HTTP service or a Python client.
