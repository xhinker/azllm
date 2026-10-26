# guard — two rescue rules for tqserve

A port of [strata-guard](https://github.com/xhinker/strata-guard)'s control-span rule and of
Strata's `reasoning_close_retry` into this fork. Pure logic: no model, no GPU, no dependencies.
Wired into `tools/tqserve.cpp` behind `--guard on`.

```bash
g++ -std=c++17 -O2 -Wall -Wextra guard/test_guard.cpp guard/guard.cpp -o /tmp/tg && /tmp/tg
```

## What it decides

A generation ends. `Guard::decide()` takes the phase's outcome and returns one of:

```
| kind            | when                                                | what phase 2 does                    |
|-----------------|-----------------------------------------------------|--------------------------------------|
| None            | finished cleanly, or a limit was reached            | nothing                              |
| CloseThinking   | the thinking block is still open, there is no answer | prompt + reasoning + close marker     |
| EscapeStop      | the answer ends in an open span                     | prompt + produced + escaped eos text  |
```

`EscapeStop` implements the control-span rule (quote, code span, code block, half-typed tag).
`CloseThinking` implements `reasoning_close_retry`. Both are bounded by `max_rescues`, and
`EscapeStop` is one-shot per burst of output: a stop with nothing written since the escape ends
the turn instead of appending another copy.

## The predicate

`inside_span(answer, window, scan)` is ported 1:1 from strata-guard's `control_span.inside_span`
and `_open_code_span`:

1. an odd number of `"` in the last `window` characters (local on purpose - global quote parity
   is "too sticky");
2. the tail ends on `|`, or on an opening backtick (trailing whitespace ignored; `>` is not an
   opener, so `` `<|im_end|>` `` is a legitimate ending);
3. a `<` with no `>` and no whitespace after it (a half-typed tag);
4. an open code fence/span anywhere in the turn (run-length fences; inline ticks only open after
   whitespace/punctuation). Rule 4 reads the whole turn - a window that starts inside a block
   would read its closing fence as an opening one - and `scan` (0 = whole turn) caps that for
   very long outputs.

The escape is the same U+200B-after-first-character convention the rest of the codebase uses for
control tokens, so the model already reads it as "you wrote the tag as text".

## Why it fits here

tqserve already runs two phases when `--reasoning-budget` ends a reply inside its thinking
(`tools/tqserve.cpp`, the `phase1`/`phase2` block). This module makes the same decision on
outcomes the budget path never sees - a clean `stop` with the thinking still open, and a `stop`
in the middle of a span.

## Wiring (done)

`tools/tqserve.cpp` calls `Guard::decide()` after a finished phase, in both the streaming and
non-streaming paths, and runs the continuation with `Decision::append` added to the prompt
(reusing the budget path's phase-2 writer pattern: intermediate writers are flushed, only the
final one is closed). Flags, all off by default:

| flag | default | meaning |
|---|---|---|
| `--guard on\|off` | `off` | enable the rescue rules |
| `--guard-rescues N` | `5` | hard bound on rescues per request |
| `--guard-window N` | `32` | characters inspected at the end of the reply (rules 1-3) |
| `--guard-scan N` | `0` | characters scanned for an open code span/block; 0 = the whole turn |

`thinking_open` is derived from the text (`text.empty() && !reasoning.empty()`), not from token
ids: tqserve splits reasoning on the think id, so a spelled-out marker never fakes a close and
no id stream is needed.

## Known limitation: EscapeStop continuation

strata-guard is a **stateful in-stream wrapper**: it withholds the stop id, injects the escaped
token into the model's own output, and re-invokes the engine from its live KV state, so the model
continues. This port runs the continuation as a **fresh re-prefill** (`prompt + produced + escaped
eos`), which is how tqserve's budget phase-2 already works. On Qwen3.8-27B the model reads the
escaped eos at the end of a re-prefilled prompt as "turn over" and emits an immediate stop, so the
`EscapeStop` rescue fires (the predicate and decision are correct) but the continuation adds no
content. `CloseThinking` is unaffected - it appends the close marker (not the eos) and reuses the
proven budget path. Making `EscapeStop` continue would need a stateful continuation in
`tools/tqserve.cpp` (append the escape to the live model state instead of re-prefilling).

## What is deliberately not ported

strata-guard's second module, `think_split.py`, undoes a reasoning->content split that a
**text-based** parser made on a *quoted* marker. tqserve splits reasoning on the think **id**,
so a spelled-out marker never matches and no wrong split occurs - the `CloseThinking` rescue
already covers the practical case (thinking left open -> close and answer). The stronger
ZWSP-after-every-character escape fallback is likewise omitted: it is a re-tokenisation check
that needs the tokenizer, and the single-ZWSP spelling is what the model already sees in its own
prompts.

## Files

```
guard.hpp        the API: inside_span(), escape_marker(), thinking_open(), Guard::decide()
guard.cpp        the predicate (rules 1-4) and the two decisions
test_guard.cpp   30 offline checks: predicate, scan, escape, both decisions, one-shot, caps
probe_template_ab.py  (unrelated) A/B probe for chat-template changes, kept here for convenience
```
