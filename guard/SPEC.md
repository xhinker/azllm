# SPEC — the guard in tqserve

Status: **wired and built.** `tools/tqserve.cpp` calls `Guard::decide()` in both the streaming
and non-streaming paths; `guard/guard.cpp` is in the `tqserve` CMake target. The notes below are
the design record; the "to-do" framing is historical.

## Flags

`--guard on|off` (default off), `--guard-rescues N` (5), `--guard-window N` (32), `--guard-scan N`
(0 = whole turn), parsed beside `--reasoning-budget`. Off by default, exactly like
`--reasoning-budget`: the rules change what a client receives, and that should be deliberate.

## What the guard needs, per finished phase

- **thinking open** = `text.empty() && !reasoning.empty()` (`tqguard::thinking_open`) - no answer
  at all, which is what a reply that ended inside its thinking looks like on the client side.
  Derived from the text, not token ids: tqserve splits reasoning on the think id, so a spelled-out
  marker never fakes a close.
- **eos marker** = `tokenizer.decode({tokenizer.eos_id()})`, cached once at startup. This is the
  text the escape wraps.

## The decision, per path

After a finished phase (the single-stream branch in both paths, and after the budget phase-2 in
the non-streaming path), `Guard::decide()` is called in a loop:

- `None` -> done.
- `CloseThinking` -> continuation with `thinking=false`, `prompt_text += reasoning + close marker`.
- `EscapeStop` -> continuation with `thinking=false`,
  `prompt_text += reasoning + text + escaped eos`.

The continuation reuses the budget path's phase-2 pattern: `max_tokens` is the remaining budget,
the produced counters are folded in, and (streaming) the writer that produced the continued-from
text is flushed while only the final writer is closed - so the reasoning is never re-sent and the
SSE stream stays one response. The loop is bounded by `Guard::decide`'s own `max_rescues` and the
one-shot rule.

## What is not ported, and why

- **think_split** (strata-guard's quoted-marker split undo): tqserve splits reasoning on the think
  **id**, not the text, so a spelled-out marker never causes a wrong split. `CloseThinking` covers
  the practical case.
- **stronger escape** (ZWSP after every character): a re-tokenisation check that needs the
  tokenizer; the single-ZWSP spelling is what the model already sees in its prompts.

## Tests

- `guard/test_guard.cpp` - 30 offline checks (predicate, scan, escape, both decisions, one-shot,
  caps), no GPU.
- Live: launch with `--guard on` and send prompts that end a reply mid-span or with the thinking
  open; a rescue appends the escaped eos (U+200B after the leading `<`) to the reply.
