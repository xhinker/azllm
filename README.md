## azllm: why this fork

azllm is a fork of [tqllm](https://github.com/turbo-tan/tqllm) set up to run Qwen3.8-27B as a daily driver for coding
agents on a single consumer GPU. The engine, weight formats and sampling are untouched; everything below is serve
layer and setup tooling.

### 1. The answer now streams when tool schemas are present

Upstream streamed `reasoning_content` correctly but buffered the answer: `update()` short-circuited text emission
whenever the request carried a non-empty `tools` array, so the reply arrived as a single `content` delta at `close()`.
Agent clients send tool schemas on every turn, so what you saw was thinking ticking along live and then the entire
answer at once, which is indistinguishable from "this server cannot stream".

The buffer had a reason. Function calls ride in the same text as prose, and `extract_tool_calls()` only runs over the
finished text, so forwarding raw deltas would leak markup into the answer. But that function keeps
`trim(text.substr(0, first))`: the answer is always a prefix of the raw text cut at the first markup marker, so
everything before that marker is already final prose. The fix streams up to the first marker and holds back only the
tail that could still grow into one, mirroring the stop-sequence hold `emit()` already applies.
The hold-back covers both the function-call and the tool-call markers, so neither can leak a partial tag into the stream.

Measured with Qwen3.8-27B-TQ3_4S + DFlash2 on one RTX 3090, `tools` non-empty:

| Request | Before | After |
|---|---|---|
| Prose answer | 1 content delta | 13 deltas, no markup in any of them |
| Prose then a function call | 1 delta | 16 deltas, `finish_reason` still `tool_calls` |
| Function call only | 0 deltas | 0 deltas, nothing to stream |
| Request without tools | unchanged | unchanged |

### 2. llama.cpp-style timing, live on the console

Timings used to exist only in the API response, so the console stayed silent for the whole request and a slow one
looked like a hang. Every request now prints a block in the shape llama.cpp's `print_timing` uses:

```
tqserve: prompt eval time =   1059.76 ms /   1215 tokens (   0.87 ms per token,  1146.49 tokens per second)
tqserve:         eval time =    851.14 ms /    282 tokens (   3.02 ms per token,   331.32 tokens per second)
tqserve:        total time =   1910.90 ms / 135763 tokens
tqserve:  draft acceptance = 0.63187 (   230 accepted /    364 drafted), mean len =  5.42
tqserve:  prompt cache =  134266 of 135481 prompt tokens reused, not prefilled
```

and while a request is still running you get a pulse every 3 s, so a long prefill is visibly moving:

```
tqserve: pp =    18432 / 38530 prompt tokens ( 47%),   1535.9 t/s,   1535.9 t/s avg
tqserve: pp =    37888 / 38530 prompt tokens ( 98%),   1194.6 t/s,   1403.2 t/s avg
tqserve: n_gen =    406, tg = 201.63 t/s, tg_3s = 135.33 t/s
```

- **`pp`** is prefill. The token count is what the GPU has finished, not what the CPU has queued: each prefill chunk
  records a CUDA event and the pulse walks them until the first one still pending, which is why the live average
  lands within about 1% of the final `prompt eval time` instead of racing ahead of it. The first rate covers the
  last 3 s, the second is the average since prefill began, and the percentage is progress toward the prompt total.
  When the interval rate drops below the average you are watching attention cost grow with context.
- **`n_gen` / `tg` / `tg_3s`** are generated tokens, average rate since generation began, and rate over the last 3 s.
- **`draft acceptance`** is the MTP / DFlash2 acceptance rate with mean accepted length. Watch this before blaming
  context length when decode looks slow: acceptance moves the rate far more than prompt size does.
- Prompt rates count **computed** tokens only. With `--prompt-cache on` a cached prefix is never prefilled, so the
  denominator is `prompt - cached`, while `usage.prompt_tokens` in the API response still reports the whole prompt
  because that is what a client is billed for.

`TQLLM_LOG_TIMINGS=0` silences all of it. Everything goes to stderr under one mutex so parallel slots cannot
interleave mid-line.

### 3. get_models.py and run_3090.sh

Two scripts, both relative to the checkout, so they work wherever you clone it.

`get_models.py` fetches a weight file plus the DFlash2 drafter and its 65,536-token draft vocabulary, into the exact
layout the launch script expects:

```sh
python3 get_models.py                     # default TQ3_4S, drafter and draft vocab into ./models
python3 get_models.py --type TQ3_4S-v2    # same weights with the fixed v22 chat template embedded
python3 get_models.py --type TQ_L         # best quality published build, TQ4 body, 14.72 GB
python3 get_models.py --type TQ3_4S-v2 --mmproj     # + the BF16 vision projector
python3 get_models.py --no-drafter --dir /data/models   # weights only, somewhere else
```

Types: `TQ3_4S` (default), `TQ3_4S-v2`, `TQ_XS`, `TQ_S`, `TQ_M`, `TQ_L`.

`run_3090.sh` serves one file on a 24 GB card and documents its own knobs in the header:

```sh
./run_3090.sh                          # TQ3_4S-v2, GPU 0, port 8086, 262K context, q4/q4 KV
GPU=1 ./run_3090.sh                    # pick the card
PORT=8090 ./run_3090.sh                # pick the port
CONTEXT=131072 ./run_3090.sh           # if weights plus KV will not fit
KV=tq3 ./run_3090.sh                   # q4/q4 (default), tq3, q8, bf16
MAX_TOKENS=32768 ./run_3090.sh         # default output cap per request (16384)
MODEL=models/Qwen3.8-27B-TQ_L.gguf TEMPLATE=models/chat_template_v22.jinja ./run_3090.sh
```

`MAX_TOKENS` is the server's default output cap (`--max-tokens`), and it binds whether or not the client asked for a
cap: `session.hpp` finishes parsing a request with `req.max_tokens = std::min(req.max_tokens, default_max_tokens)`, so
a request for 100000 tokens against a server started with `--max-tokens 4096` gets 4096 and nothing in the response says
so. Thinking is on by default and bills against that same budget, so the turn simply ends early with
`finish_reason: length` and the model reads as one that stops mid-sentence. The default was 4096 until 2026-10-15;
16384 leaves room for a long think plus a long answer, and against the 262144 default context it still leaves 245760
tokens for the prompt.

The script exports `CUDA_VISIBLE_DEVICES` instead of prefixing the command with `env`, and that is deliberate: a
`~/.local/bin/env` PATH shim (installed by uv) shadows `/usr/bin/env` on many setups, so
`env CUDA_VISIBLE_DEVICES=1 ./build/tqserve ...` runs the shim, which edits `PATH` and exits 0. The server never
starts and prints nothing at all.

### 4. Fastest path from clone to first token

```sh
git clone https://github.com/xhinker/azllm.git && cd azllm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tqserve -j"$(nproc)"
python3 get_models.py --type TQ3_4S-v2      # ~14.5 GB: weights, drafter, draft vocab
./run_3090.sh                               # ready when it prints "listening on 0.0.0.0:8086"
```

Then from anywhere else:

```sh
curl -s localhost:8086/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "tqllm",
  "messages": [{"role": "user", "content": "Name one prime number."}],
  "max_tokens": 64
}'
```

Point any OpenAI-compatible client at `http://localhost:8086/v1` with model name `tqllm`. Four things that save time:

- **Give `max_tokens` room.** Reasoning is on by default (`--reasoning on` in the launch script) and it bills against
  the same budget, so `max_tokens: 24` can return an empty `content` with every token spent on `reasoning_content`.
  Per request, set `"chat_template_kwargs": {"enable_thinking": false}` or `"reasoning_budget_tokens": 0` to skip
  thinking, or `"reasoning_effort"` to dial it down.
- **The silence after the banner is normal.** Loading 14.5 GB of weights takes 20-40 s with no output between the
  `loading model` line and `listening`.
- **Budget the memory.** A 24 GB card fits one 27B at 262K context with `q4/q4` KV, about 23 GB resident. Drop to
  `CONTEXT=131072` if you want headroom; a 32 GB card takes the same config with ~8 GB spare.
- **Know which template your file carries.** `TQ3_4S-v2` embeds the fixed v22 template and needs nothing. `TQ_XS`,
  `TQ_S`, `TQ_M` and `TQ_L` embed the official template, which fails on tool arguments sent as a JSON string, so
  pass `TEMPLATE=models/chat_template_v22.jinja`.

### 5. Measured on consumer cards (2026-10-14)

Same engine build, same 262,144-token context, `q4/q4` KV, DFlash2 drafter, `--prompt-cache on`, temperature 0.
Prompts are random-word filler at four sizes, generated fresh each run so nothing hits the prompt cache unless the
row says so. Repeats: 3, 3, 2 and 1; prefill varied under 2% within a size, decode up to +-20%.

Prefill and decode in tokens/s (cold cache):

| Prompt | TQ_L, RTX 3090 Ti | TQ3_4S-v2, RTX 3090 Ti | TQ_L, RTX 5090 |
|---|---|---|---|
| short, ~135 tok | 566 / 211 | 650 / 207 | **1,319 / 333** |
| medium, ~2,350 tok | 1,481 / 139 | 1,453 / 128 | **2,848 / 241** |
| long, ~36,550 tok | 1,403 / 118 | 1,363 / 166 | **2,826 / 233** |
| xlong, ~145,900 tok | 941 / 127 | 922 / 140 | **2,078 / 243** |

End to end, which is what you actually wait for:

| Request | TQ_L, RTX 3090 Ti | TQ_L, RTX 5090 |
|---|---|---|
| short, 128 tokens out | 0.8 s | **0.5 s** |
| medium, 256 out | 3.3 s | **1.9 s** |
| long, 256 out | 28.2 s | **14.0 s** |
| xlong, 128 out | 156.3 s | **70.8 s** |
| long prompt, warm cache (first token) | 185 ms | **92 ms** |

What the numbers say:

- **The 5090 is a clean ~2x on everything**: prefill 1.9-2.3x, decode 1.6-2.0x, consistently across all four sizes.
- **The quant choice does not move speed.** TQ_L and TQ3_4S-v2 differ by 220 MB and by 2-3% in prefill, which is
  noise. Their decode columns look different only because draft acceptance differed (0.24-0.32 vs 0.40-0.44);
  acceptance is a property of the text generated, not of the weight format. Pick between them on output quality.
- **Small prompts prefill badly and it does not matter**: 566-650 tok/s at 135 tokens, because one short sequence
  cannot fill a GEMM batch. That is 0.2 s of wall clock.
- **The long-context cliff is the same shape on both cards**: 36K to 146K costs 33% of prefill on the 3090 Ti and
  26% on the 5090. Quadratic attention scales, it does not disappear.
- **The prompt cache is the biggest lever available.** One 36,564-token prompt, twice: 26.2 s cold, 185 ms warm on
  the 3090 Ti (141x), 13.2 s cold, 92 ms warm on the 5090. Anything with a stable prefix stops paying prefill.
- **Memory**: 23,006 MiB of 24,564 on the 3090 Ti, 23,644 MiB of 32,607 on the 5090. Same config, but only the 5090
  has ~8 GB left over for a bigger context, a second slot, or a second resident model.

Caveats worth repeating: single slot, so no concurrency was measured; reasoning was on, so decode counts thinking
tokens as output; clocks were sustained rather than boost (3090 Ti at 1875 of 2115 MHz and 396 W, 5090 at 2782 MHz
and 439 W); and the 146K prompt used 56% of the configured context.

### 6. Model and runtime metadata on `/v1/models`, `/props` and `/health`

Upstream's `/v1/models` answered with `{"id","object","owned_by"}` and nothing else, so a client could not ask the
server how much context it actually has. llama.cpp answers that question and clients read the answer: AZ PAL takes
`data[].meta.n_ctx` and, failing that, `max_context` from `/health` at the server root. Without one of those a client
falls back to whatever number someone typed into a config file, which is how a profile ends up carrying another
machine's context window. tqserve now reports it, under llama.cpp's field names:

```sh
curl -s localhost:8086/v1/models | python3 -m json.tool
```

```json
{"id": "tqllm", "object": "model", "owned_by": "tqllm",
 "meta": {"vocab_type": 2, "n_vocab": 248320, "n_ctx": 262144, "n_ctx_train": 262144,
          "n_embd": 5120, "n_params": 27320697856, "size": 14721469184, "ftype": "TQ4 - 4.12 bpw"}}
```

| Field | Meaning |
| --- | --- |
| `n_ctx` | positions ONE request gets: `--context` divided by `--np`, cut further if the memory fit said so |
| `n_ctx_train` | the checkpoint's `qwen35.context_length`, 262144 for Qwen3.8-27B |
| `n_params` | element count of the tensor directory; matches llama.cpp on the same checkpoint to the unit |
| `size` | the weight file on disk, in bytes |
| `ftype` | measured from the bytes that store the TQ body, NOT from `general.file_type`: the TQ converter stamps that key 15, which in llama.cpp's enum claims Q4_K_M for a file whose tensors are type 300 |

`GET /props` carries `n_ctx_train` and the vocabulary's `bos` / `eos` / `eot` token ids. Both endpoints, and `/health`,
also return a `tqllm` object holding the facts upstream clients have no field name for, so nothing can mistake them for
llama.cpp fields: `slots`, `max_tokens`, `kv`, `kv_store`, `draft`, `reasoning`, `reasoning_effort`, `weight_types` and
`prompt_cache_active`.

Watch that last one. Prefix reuse needs a single slot, so `--np 2` costs a client every cached prefix and the log never
mentions it; `prompt_cache_active` reads `false` the moment that happens. `/health` publishes `max_context`, `slots`,
`max_tokens` and `prompt_cache_active` for probes that do not read `/v1/models` at all.

All of it is read once at startup from the GGUF header in `tools/serve/model_info.hpp`. No engine file is involved, so
the fork still merges upstream without conflict.

### 7. A prompt cache that survives a rewritten history

`--prompt-cache` keeps the previous request's KV so the next one only prefills what is new, by checking whether the
prompt it was handed starts with the tokens it already read. In a normal chat every turn only adds to the end, so that
is true every time and reuse runs 95-99.9%.

That check is yes-or-no, and that is the problem. The engine could resume in exactly two places: the end of what it
just read, or a snapshot 64 tokens before that end. A client that *edits its own history* breaks both at once. An
agent that trims a long tool result down to a line before saving the conversation, a memory system that summarises old
turns, a client that reorders a preamble: all of them send back a prompt that is 99% identical to one already
computed and differs somewhere in the middle. Neither resume point is usable then, so the outcome is all or nothing -
nothing reused, the whole conversation read again, tens of seconds before the first token on a 150K conversation.
Almost every token in that re-read was still correct.

**How to reproduce it.** The server prints what it reused on every request:

    tqllm: prompt cache: 145000 of 145300 prompt tokens reused

Start a server with `--prompt-cache on`, send a long prompt, then send one that extends it unchanged - the second
line shows near-total reuse. Now send that prompt again with one sentence changed near the beginning. The line shows
`0 of N` and the prefill runs over everything. Nothing failed; the cache just had nowhere to resume.

**What was added.** `--prompt-cache-step N` writes a bookmark every N prompt tokens while prefilling and keeps the
last eight. On the next request the engine finds the real shared prefix by binary search and resumes from the deepest
bookmark at or before it. A bookmark holds the recurrent and short-conv state, which is the part a hybrid model
cannot recover from nothing; the per-token KV of the attention layers is already in the pool and needs no copy. The
server prints the size of one bookmark the first time it writes one.

Why bookmarks and not just the search: the attention layers keep one row per token, so a matching prefix is correct
wherever it happens to stop. The gated-deltanet layers keep a running summary, so "the state at token 128,000" only
exists if somebody saved it. Without bookmarks, knowing exactly where the two prompts diverge tells you nothing you
can act on.

| The next prompt | Before | With `--prompt-cache-step 16384` |
|---|---|---|
| Adds to the end (every ordinary turn) | reuses everything | same path, same speed |
| Re-renders only the previous answer | reuses up to the tail snapshot | unchanged |
| Changes something in the middle (a trimmed tool result) | nothing reused, whole conversation re-read | re-reads at most one step, plus the changed tail |
| Replaces the oldest turns with a summary | nothing reused | still nothing: the change is at the head, so no bookmark sits before it |

Off by default, and passing nothing leaves the code path untouched. Through the launch script:

    MODEL=models/Qwen3.8-27B-TQ_L.gguf PC_STEP=16384 GPU=2 PORT=8086 ./run_3090.sh

The mechanism is read from the source and the arithmetic follows from it, but no card has measured this yet: the
table says what the code does, not what a benchmark returned.

### 8. A guard that rescues a reply that stopped mid-span

A reply can stop in the middle of something it was still writing - a quote left open, a code fence never closed, a
tag half typed - because the model emitted the turn-end token as if it were text. The turn ends where the model meant
to keep going. Strata ships a rescue for exactly this (strata-guard); this is that logic, ported to the engine and
wired into `tqserve` behind a flag.

Two rules, both off by default:

- **Close thinking.** The reply ended with its thinking block still open, so the client got reasoning and no answer.
  The guard closes the block and lets the model answer - the same two-phase machinery the reasoning budget already
  uses.
- **Escape stop.** The answer ends inside an unterminated quote, code span, code block or half-typed tag. The guard
  appends the escaped spelling of the turn-end token - a zero-width space after its first character, so it cannot
  re-tokenise into the reserved id - and continues, so the turn can finish.

The span test is the ported predicate: an odd number of quotes in the tail, a trailing `|` or an opening backtick, a
`<` with no `>` after it, or an open code fence anywhere in the turn. It is unit-tested offline
(`guard/test_guard.cpp`, 30 checks), and each decision is bounded by a per-request rescue budget.

| The reply ended... | Guard off | With `--guard on` |
|---|---|---|
| ...with an open quote / span / fence / half-typed tag | as-is, mid-span | escape appended, the turn continues |
| ...with its thinking block open (no answer) | reasoning only, no answer | block closed, the model answers |
| ...normally | unchanged | unchanged (the guard never fires) |

Off by default; without the flag the path is byte-identical to before. The server logs each rescue. Through the
launch script:

    GUARD=on GPU=2 PORT=8086 ./run_3090.sh

One honest caveat: the continuation re-prefills the reply plus the escape rather than continuing from live state, so
on some models the turn repeats instead of finishing the span. The detection and the rescue are correct; making the
continuation stateful is the open follow-up.

### Updates

Newest first. Everything below is serve layer or tooling; the engine, weight formats and sampling stay upstream -
with two exceptions, 2026-10-20 (§7) and 2026-10-26 (§8).

**2026-10-26: a guard that rescues a reply that stopped mid-span (§8)**

- `--guard on` enables two rescue rules ported from strata-guard: close an open thinking block (the reply got
  reasoning but no answer), and escape-stop a reply that ended mid-span (open quote / code span / fence / half-typed
  tag) by appending the escaped turn-end token and continuing. Off by default; without the flag the path is
  byte-identical to before. `run_3090.sh` accepts it as `GUARD` (plus `GUARD_RESCUES` / `GUARD_WINDOW` / `GUARD_SCAN`).
- New `guard/` module (the span predicate + the decision, unit-tested offline) wired into `tools/tqserve.cpp`;
  `Tokenizer::piece()` added so the text of the special turn-end token can be read (`decode()` drops specials). The
  server logs each rescue.

**2026-10-20: the prompt cache can resume from the middle of a conversation (§7)**

- `--prompt-cache-step N` writes state bookmarks while prefilling, so a prompt that differs mid-history - a trimmed
  tool result, a re-rendered preamble - resumes from the nearest bookmark instead of re-reading the whole
  conversation. Off by default; without the flag the path is byte-identical to before. `run_3090.sh` accepts it as
  `PC_STEP`.
- First change in the fork that touches the engine rather than the serve layer or the tooling: `src/qwen35.cpp`,
  `src/options.cpp`, `include/tqllm/options.hpp`. Carry it explicitly when merging upstream.

**2026-10-16: the tool-stream hold-back also covers the tool-call marker**

- The hold-back that keeps tool-call markup out of the streamed answer now stops at the tool-call marker as well as the
  function-call marker, matching where `extract_tool_calls()` cuts. Prose before a tool call streams, and a tail that
  could still grow into either marker is withheld, so no partial tag leaks into the deltas (§1). `test-serve` gained a
  regression test for it.

**2026-10-15: model and runtime metadata on the API; a larger default output cap**

- `GET /v1/models`, `GET /v1/models/<name>`, `GET /props` and `GET /health` publish the model and runtime numbers under
  llama.cpp's field names, so a client reads its context window from the server rather than from a config file (§6).
  `n_params` computed here agrees with llama.cpp on the same checkpoint exactly: 27,320,697,856.
- `run_3090.sh` raises its default `MAX_TOKENS` from 4096 to 16384. A request above the server default is clamped to it
  without an error, and thinking bills against the same budget, so 4096 surfaced as turns ending mid-sentence with
  `finish_reason: length` (§3).

**2026-10-14: streaming with tool schemas, live timing, launch tooling**

- The answer now streams when a request carries `tools`; upstream buffered it until `close()` (§1).
- llama.cpp-style timing on the console: computed tokens, GPU progress, live pulse (§2).
- `get_models.py` and `run_3090.sh` for a single-GPU box (§3), and measured numbers on consumer cards for TQ3_4S, TQ_S
  and TQ_L (§5).


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tqllm-logo-dark.png">
    <img src="docs/assets/tqllm-logo.png" alt="tqllm" width="360">
  </picture>
</p>

# tqLLM

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-davidytan2000-BD5FFF?logo=buymeacoffee&logoColor=white)](https://www.buymeacoffee.com/davidytan2000) [![GoFundMe](https://img.shields.io/badge/GoFundMe-support%20TQLLM-00B964?logo=gofundme&logoColor=white)](https://gofund.me/0f79d4274)

C++/CUDA inference for supported Qwen models and TurboQuant GGUF weights,
with an OpenAI-compatible HTTP server.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for building, testing and where new models, weight formats and KV caches plug in, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a map of the code.

## Credits

tqLLM re-implements ideas from the work below (no code unless listed in NOTICE). Reference = their idea, method or
weights; improvement = their idea, extended or re-implemented by us.

| Piece | Credit |
|---|---|
| Qwen3.8-27B weights, MTP head, chat template | Qwen team, Alibaba ([Qwen](https://huggingface.co/Qwen)) |
| GGUF format, K-quant / IQ block layouts, reference engine; base of our llama.cpp-tq3 fork | ggml authors, [llama.cpp](https://github.com/ggml-org/llama.cpp), Georgi Gerganov |
| DFlash / DFlash2 drafting method and drafter weights (served here in TQ3 / TQ4 with a 65,536-token draft vocabulary) | Jian Chen, Yesheng Liang, Zhijian Liu, [z-lab](https://github.com/z-lab/dflash) ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) |
| Rotation-based quantisation, the lineage of the TQ3 / TQ4 weight formats and the tq3 KV cache | Google Research, [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) |
| turbo2 / turbo3 / turbo4 KV types, REFRACT quality metric | Tom Turney (TheTom), [turboquant_plus](https://github.com/TheTom/turboquant_plus) |
| Trellis-coded KV idea (tcq codecs; codebooks and kernels ours) | spiritbuun (buun), [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) |
| Learned rotations for quantisation | Avarok, TorQuant |
| Fixed chat template (v22, embedded in our files) | froggeric, [Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) |
| FlashAttention tiling | Tri Dao et al., [flash-attention](https://github.com/Dao-AILab/flash-attention) |
| Speculative sampling | Leviathan, Kalman, Matias ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192)); Chen et al. ([arXiv 2302.01318](https://arxiv.org/abs/2302.01318)) |
| Token-tree verification (our hybrid Gated DeltaNet tree kernels build on it) | SpecInfer, Miao et al. ([arXiv 2305.09781](https://arxiv.org/abs/2305.09781)); [Medusa](https://github.com/FasterDecoding/Medusa), Cai et al.; [EAGLE](https://github.com/SafeAILab/EAGLE), Li et al. |
| Prompt-lookup drafting (`--lookup`) | Apoorv Saxena, [prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) |
| Comparison baselines | [Unsloth](https://huggingface.co/unsloth) UD quants; [HyperQwen](https://github.com/syv-ai/HyperQwen); [ExLlamaV3](https://github.com/turboderp-org/exllamav3), turboderp; [vLLM](https://github.com/vllm-project/vllm); Strata, Niko1221; LocalMaxxing |

### Testers

Thank you to the testers from X who ran tqLLM on their own hardware and reported back: 0x443, Ricky Spanish,
lackonendes and GandalfOne.

## Get the image

The prebuilt image is public on the GitHub container registry and on Docker Hub. One multi-platform tag covers x86-64
(RTX 30 / 40 / 50) and arm64 (GB10 / DGX Spark); `docker pull` picks the right one.

```sh
docker pull ghcr.io/turbo-tan/tqllm:1.0.0
# or
docker pull ytan2000/tqllm:1.0.0
```

`latest` always points to the newest release.

## Docker quick start

Requires Linux, an NVIDIA GPU, Docker Compose, and NVIDIA Container Toolkit.
One image tag covers x86-64 PCs with RTX 30, 40 or 50 series GPUs (CUDA architectures 86, 89, 120) and arm64
GB10 / DGX Spark systems (121); `docker pull` picks the right platform and the engine picks its kernels for your GPU at
start. Apple Metal is not supported.

Place a supported model at `models/model.gguf` (the tokenizer is read from the GGUF). Supported: Qwen3.5 /
Qwen3.8 dense models (hybrid attention + Gated DeltaNet) and qwen35moe mixture-of-experts models such as MiniCPM-V 4.7
35B-A3B (text; `--moe-host-mb N` keeps the least-used experts in system RAM when the file is larger than the GPU), in
TQ3, TQ4, TQ3_4S, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q4_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, Q2_K, IQ2_S, IQ2_XS, IQ2_XXS,
IQ1_S, F16 and BF16 tensors (so Q4_K_M, Q5_K_M, Q6_K, Q8_0 and 2-bit Unsloth UD files load). IQ1_M does not load yet.

Then run:

```sh
docker compose up            # uses the prebuilt image
docker compose up --build    # or build the image from this source
```

To use an existing models directory and different filenames:

```sh
MODEL_DIR=/path/to/models MODEL_NAME=your-model.gguf docker compose up --build
```

Models are mounted read-only. The API listens at `http://localhost:8086`.
Context defaults to 16384; increase CONTEXT only when the model fits your GPU.

```sh
curl http://localhost:8086/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain a hash map."}],"temperature":0,"max_tokens":128}'
```

## tqllm launch: a coding agent on your own GPU

`tools/launch/tqllm.py` starts a local server and runs a coding agent against it, like `ollama launch`. It needs
Python 3, Docker with the NVIDIA Container Toolkit (or a native `tqserve` build) and the agent itself installed.

```sh
python3 tools/launch/tqllm.py --list                    # model catalog (local copies are marked)
python3 tools/launch/tqllm.py launch opencode           # the default model for this GPU, downloaded if missing
python3 tools/launch/tqllm.py launch codex --model qwen3.8-27b-xs
python3 tools/launch/tqllm.py launch claude --model models/Qwen3.8-27B-TQ_S.gguf \
  --drafter models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models/mtp-vocab/atx_65536.txt
python3 tools/launch/tqllm.py serve --model qwen3.8-27b  # just the server, until Ctrl+C
```

- **Agents:** `opencode`, `codex` and `claude` (Claude Code, through a built-in Anthropic Messages to OpenAI bridge).
  Arguments after `--` go to the agent, e.g. `launch opencode -- run "fix the failing test"`.
- **Models:** `--model` takes a GGUF file, a catalog name from `tools/launch/models.json`, or nothing (the default
  for the GPU's memory). Catalog files are looked up in `./models`, `~/models` and `$TQLLM_MODELS`; missing ones are
  offered for download from Hugging Face with a size and sha256 check (`--yes` skips the question).
- **Server:** the Docker image by default, or `--native build/tqserve`. Context defaults to 65536 because agent
  prompts are long; `--context`, `--port`, `--reasoning` and `--server-arg` (repeatable) pass through. A server already
  answering on the port is reused and left running; one the command started is stopped when the agent exits or the
  command gets Ctrl+C / SIGTERM.
- The agents' own configuration files are never written; each gets its provider through its environment or command
  line.

## TQ formats: TQ3_4S, TQ3 and TQ4

All three rotate the weights before quantising (the TurboQuant idea: a rotation spreads outliers so a small codebook fits
every block), then store codes plus scales. They differ in block size, scales and where they run:

| Format | Block | Scales | Bits per weight | Runs in |
|---|---|---|---:|---|
| TQ3_4S (GGUF type 46) | 32 weights, 32-point rotation, 8-level code | four 8-bit scales (one per 8 weights) | 4.0 | tqLLM and [llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) |
| TQ3 (type 301) | 128 weights, 128-point rotation, 8-level code | one FP16 scale per 128 | 3.125 | tqLLM only |
| TQ4 (type 300) | 128 weights, 128-point rotation, 16-level code | one FP16 scale per 128 | 4.125 | tqLLM only |

TQ3_4S spends a quarter of its bits on scales; TQ3 and TQ4 rotate over a 4x longer block, so one scale per 128 weights
is enough and the bits go into the codes instead. That is how the 27B fits in 11.85 GB as TQ3 (TQ_XS) against 14.5 GB as
TQ3_4S-v2, and why TQ4 at about the same size as TQ3_4S is much closer to the original (KLD vs BF16 0.022 for TQ4_L).
The release files `Qwen3.8-27B-TQ_XS.gguf` (11.85 GB), `TQ_S` (12.55 GB) and `TQ4_L` (14.97 GB) are built from these
formats. Type ids 300
and 301 are tqLLM's own (above the ggml range), so these files need tqLLM; llama.cpp cannot load them.

## Fast decoding: about 250 tok/s on an RTX 3090

tqLLM speculates with a DFlash2 draft model: the drafter proposes several tokens, the 27B model checks them in one pass,
and commits the ones it would have produced itself. Greedy output is identical, token for token, to decoding without
the drafter: plain decoding and every verify (tree, chain, MTP) share one attention arithmetic, checked by
`test-tree`, `test-dflash-exact` and `test-native-mtp` and on long prompts up to 51K tokens. Download into `./models`:

| file | Hugging Face repo |
|---|---|
| `Qwen3.8-27B-TQ3_4S-v2.gguf`, the 27B model (14.5 GB) | `YTan2000/Qwen3.8-27B-TQ3_4S` |
| `drafters/Qwen3.8-27B-DFlash2-TQ3.gguf`, the DFlash2 drafter (0.79 GB) | `YTan2000/Qwen3.8-27B-TQ` |
| `mtp-vocab/atx_65536.txt`, the drafter's token list | `YTan2000/Qwen3.8-27B-TQ` |

```sh
hf download YTan2000/Qwen3.8-27B-TQ3_4S Qwen3.8-27B-TQ3_4S-v2.gguf --local-dir models
hf download YTan2000/Qwen3.8-27B-TQ drafters/Qwen3.8-27B-DFlash2-TQ3.gguf mtp-vocab/atx_65536.txt --local-dir models
```

RTX 3090, 32K context, thinking on:

```sh
docker run --rm --gpus all -p 127.0.0.1:8086:8086 -v "$PWD/models:/models:ro" \
  -e MODEL_FILE=/models/Qwen3.8-27B-TQ3_4S-v2.gguf \
  -e DRAFT_MODEL_FILE=/models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
  -e DRAFT_VOCAB_FILE=/models/mtp-vocab/atx_65536.txt \
  -e CONTEXT=32768 -e REASONING=on -e MAX_TOKENS=4096 \
  ghcr.io/turbo-tan/tqllm:1.0.0
```

With Compose, set the same names: `DRAFT_NAME=drafters/Qwen3.8-27B-DFlash2-TQ3.gguf
VOCAB_NAME=mtp-vocab/atx_65536.txt MODEL_NAME=Qwen3.8-27B-TQ3_4S-v2.gguf CONTEXT=32768 REASONING=on docker compose up`.

What to expect: on the LocalMaxxing reasoning prompt (32K context, 320 output tokens) an RTX 3090 runs at 240-270 tok/s
with this drafter (TQ3 and Q4_K_M builds measure the same). The server reports tok/s for every
request. Speed depends on how often the drafter's guesses are accepted, so it varies with the text: predictable text
(reasoning, prose) runs fastest, code and long contexts slower.

Long prompts: `--draft-retrieve 32` lets the drafter also read 32 far passages of the prompt (RTX 3090, 64K context:
31K / 40K / 51K-token prompts 158 / 143 / 137 -> 176 / 156 / 148 tok/s, short prompts -2 %). It keeps the drafter's
cache for the whole context (about 20 KB per token), so it is off by default.

No drafter file: `--mtp` drafts with the model's own prediction head (one user, greedy; same output as plain decoding):
155 tok/s at 1.6K and 125 tok/s at 51K tokens on an RTX 3090, against 57 and 47 without drafting.

### Recommended settings

| GPU | model | `CONTEXT` | `KV_FORMAT` | drafter | measured decode (tok/s) |
|---|---|---|---|---|---|
| RTX 3090 / 4090 (24 GB) | `Qwen3.8-27B-TQ3_4S-v2.gguf` | 32768 | default (bf16) | yes | 131-155 on summaries, 256 on LocalMaxxing |
| RTX 5060 Ti / 4060 Ti (16 GB) | `Qwen3.8-27B-TQ3_4S-v2.gguf` | 16384 | `tq3` | yes | 77-95 on summaries |
| GB10 / DGX Spark (128 GB) | `Qwen3.8-27B-TQ3_4S-v2.gguf` | 131072 or more | default (bf16) | yes | 43-54 on summaries, 79 on LocalMaxxing |

- `KV_FORMAT=tq3` stores the KV cache at about 3 bits per value instead of 16. Use it when memory is tight or the context
  is long; on 24 GB at 32K the default is fine.
- The memory fit runs at start: if `CONTEXT` does not fit, tqserve moves the token embeddings and LM head to host
  memory first, then shrinks the context, and logs each step. Pass `-fitt N` to keep N MiB of the GPU free (default
  768, 256 on WSL).
- `REASONING=on` for thinking tasks, with `MAX_TOKENS` large enough for the thinking plus the answer (4096 or more).
- `PARALLEL=N` serves N requests at once; the context is split evenly between them.

## Long context on a 16 GB card (TqCache)

Ask for more context than the GPU holds and the memory fit keeps the oldest KV positions in system RAM (exact: the
output is the same as with all KV on the GPU) instead of refusing the request. Reading that part over PCIe costs decode
speed, so use it for prompts past what the GPU holds; below that, the default context is faster.

RTX 5060 Ti 16 GB, Qwen3.8-27B-TQ_XS.gguf + DFlash2-TQ3, tq3 KV, decode tok/s. Default context: the memory fit caps the
context at 204,800 on this card and keeps all KV on the GPU. `--context 262144`: TqCache keeps the oldest positions in RAM.

| prompt | default context (204,800, all KV on the GPU) | `--context 262144` (TqCache) |
|---|---|---|
| 200K | 113 | 70 (81,920 of its positions read from RAM) |
| 224K | refused (longer than the context) | 42 |
| 248K | refused | 32 |

The fit reserves RAM for the whole requested context up front, so with `--context 262144` even a prompt that would fit
on the GPU reads part of its KV over PCIe (the 200K row). Ask for more context than the default only when the prompts
need it.

## Options

`tqserve --help` prints the same list. In Docker, the variables below set the common ones and any other option goes
after the image name (`docker run ... ghcr.io/turbo-tan/tqllm:<tag> --prefill-int8 on`).

Docker variables: `MODEL_FILE`, `TOKENIZER_FILE`, `DRAFT_MODEL_FILE`, `DRAFT_VOCAB_FILE`, `KV_FORMAT`, `CONTEXT`
(16384), `PARALLEL` (1), `MAX_TOKENS` (1024), `REASONING` (off).

### Model and serving

| Option | Default | What it does |
|---|---|---|
| `--model FILE` / `--model-dir DIR` | | the GGUF (or a safetensors directory) to serve |
| `--tokenizer FILE` | the GGUF itself | only for safetensors or a deliberate override |
| `--host IPv4`, `--port N` | 127.0.0.1, 8086 | where the API listens |
| `--context N` | 2048 | context per server, split evenly across `--np` slots; the memory fit may lower it |
| `--max-tokens N` | 512 | default output cap per request |
| `--np N` | 1 | requests served at once (batched verify); `GET`/`POST /admin/slots` reads or changes the live limit |
| `--name NAME`, `--system TEXT` | | model name in the API; a default system prompt |
| `--api-key KEY` / `--api-key-file FILE` | | require a bearer key |
| `--mmproj FILE [--image-min-tokens N] [--image-max-tokens N]` | 8, 4096 | vision projector for image input |
| `--repeat-limit N` | 256 | end a request after N identical tokens in a row (0 off) |

### Reasoning and chat template

| Option | Default | What it does |
|---|---|---|
| `--reasoning on\|off` | off | thinking mode |
| `--reasoning-effort low\|medium\|xhigh` | | effort hint for the template |
| `--reasoning-budget N` (alias `--thinking-budget`) | -1 | thinking token budget (-1 unlimited, 0 end now; llama.cpp semantics) |
| `--reasoning-budget-message TEXT` | | text inserted when the budget ends the thinking |
| `--reasoning-answer-reserve N` | 8192 | tokens kept for the answer after a budgeted think (0 = budget counts inside max_tokens) |
| `--jinja` / `--no-jinja` | jinja | the model's Jinja chat template (llama.cpp's engine) or the native Qwen layout |
| `--chat-template-file FILE`, `--chat-template-kwargs JSON` | | override the template / pass template variables |

### Speculative decoding (drafter)

| Option | Default | What it does |
|---|---|---|
| `-md FILE` | | DFlash2 drafter GGUF (exact: the output equals decoding without it) |
| `--draft-vocab FILE` | | the drafter's token list (65,536 ids) |
| `--draft-max N` | 7 | drafted tokens per step (1-15); hybrid Qwen3.5 / 3.8 models verify at most 8 columns, so 7 is the effective maximum there |
| `--no-tree` | tree on | verify a single draft chain instead of a token tree |
| `--draft-window N` | GGUF value | drafter attention window over the context (0 = whole context) |
| `--lookup L [--lookup-first K]` | off | prompt-lookup drafts (copies of earlier text) next to the tree; shares the verify budget |

### KV cache and long context (TqCache)

| Option | Default | What it does |
|---|---|---|
| `--kv FORMAT` / `-ctk K -ctv V` | bf16 | KV cache format; `--list-kv` lists them (tq3 = 3-bit, trellis tcq3 / tcq2, q8, q4, splits) |
| `--kv-host N` | memory fit | KV positions kept in system RAM (exact); by default the fit moves only what does not fit on the GPU |
| `--kv-stage on\|off` | on | stream the RAM part back to the GPU layer by layer while decoding |
| `--kv-compact R [--kv-compact-at idle\|turn]` | off, idle | compress old KV R-fold (not exact; one slot, tq3 or trellis KV) |
| `--prompt-cache on\|off` | off | keep the previous turn's KV so a follow-up does not re-read the document (one slot; on with `--kv-compact`) |
| `--prompt-cache-step N` | off | with `--prompt-cache`: bookmark the state every N prompt tokens so a prompt that differs mid-history resumes from the nearest bookmark instead of reading everything again (§7) |
| `--kv-store flat` | flat | KV storage policy (only flat is served today) |
| `--rope-scaling none\|linear\|yarn --rope-scale F [--yarn-orig-ctx N]` | none | context past the trained length (llama.cpp names) |

### Memory fit and prefill

| Option | Default | What it does |
|---|---|---|
| `-fit on\|off` | on | load, measure, and step down until the margin is free, in this order: token embeddings and the LM head to host memory, smaller prefill buffers, a smaller KV format (when `--kv` is not set), prefill weight blocks, the KV tail to system RAM, then a shorter context |
| `-fitt MiB` | 768 (256 under WSL) | free memory to leave on the GPU (64 MiB floor) |
| `-fitc N` | 4096 | smallest context the fit may choose (with `--np`: the smallest context per slot) |
| `-fitl on\|off` | on | on: when the requested context / users do not fit at full settings, pick a compact KV cache (q4/tcq2), smaller prefill buffers or weight blocks, or a shorter context, and say so at start-up (and in `/health`); off: refuse instead |
| `--gemm-rows N` | 0 (fit) | prefill FP16 weight blocks of N rows (less VRAM); the memory fit picks 2048 when needed and says so |
| `--mtp` | off | draft with the model's own MTP (nextn) head instead of `-md` (text-only, greedy, `--np 1`; default depth 4, `--draft-max N`) |
| `--draft-retrieve N` | 0 (off) | with `-md`: N far chunks of 16 positions in the drafter window; long prompts get faster (32: +8-12 % at 31K-51K on a 3090) but the drafter keeps K/V for the whole context (~20 KB / token: 1.25 GiB at 64K, 3.9 GiB at 200K), so turn it on when memory allows |
| `--mtp-from N` | 0 | with `-md` and `--mtp`: prompts of N or more tokens draft with the MTP head (`--mtp-depth D`, default 4), shorter ones with DFlash2 |
| `--slot-reserve MiB` | 128 | free memory per parallel slot after loading; an `--np` that does not fit is refused at startup with the largest that does |
| `--graph-cache N` | 256 | captured CUDA graphs kept (~12 MB each); least recently used are evicted, also when free memory runs low |
| `-ub N` | automatic | prefill chunk (ubatch) size |
| `--prefill-fp8 on\|off` | off | FP8 prefill GEMMs (sm_89+): about +40 % prefill on GB10 but KLD ~0.01 vs FP16 prefill, so opt-in; slower on 16 GB cards |
| `--prefill-int8 on\|off` | on for RTX 30 / 40 / 50 | int8 prefill GEMMs (RTX 50: +11-13 % prefill on a 5060 Ti; RTX 40 by analogy) |

### Memory fit: what it does and how to override it

You normally pass only `--context` (and `--np`). If that does not fit at full settings, the server picks what fits and
prints one line, for example:

```
tqserve: fit: to fit 1 slot(s) x 131072 in GPU memory: KV cache q4/tcq2 (asked bf16), prefill buffers -ub 512,
prefill FP16 weight blocks of 2048 rows (choose yourself with --kv, -ub, --gemm-rows or a smaller --context)
```

`GET /health` reports the same in its `fit` field. Your own settings always win; examples:

| you want | add |
|---|---|
| full-quality BF16 KV, accept a shorter context if needed | `--kv bf16` |
| a specific compact cache | `--kv q4/tcq2` (trellis V, lowest KLD of the compact caches), `--kv q8/q8`, `--kv tq3` (`--list-kv`) |
| bigger prompt buffers (faster prefill, more memory) | `-ub 2048` |
| full weight matrices for prefill (faster, more memory) | `--gemm-rows 0` |
| exactly what you asked for, or an error | `-fitl off` |
| several users at a long context (e.g. 4 x 32K) | `--np 4 --context 131072` (the context is split across `--np`) |
| keep more free memory for other programs | `-fitt 2048` (MiB left free) |

### Diagnostics

| Option / variable | What it does |
|---|---|
| `--list-kv`, `--list-flags` | KV formats; every environment flag with its status |
| `TQLLM_DEBUG=key,...` | memlog, profile, steps, stageprof, matvec, tune, prefill, draft, tree_chain, tf_dump:FILE, fakeq_k:N, fakeq_v:N |
| `TQLLM_MATVEC_RULE=BITS:KERNEL[:PF],...` | override the decode matvec kernel per format (e.g. `3:legacy:0`) |

## Performance

Measured on the model and drafter above (`Qwen3.8-27B-TQ3_4S-v2.gguf` + `Qwen3.8-27B-DFlash2-TQ3.gguf`), engine built
from the named commit, tq3 KV cache, greedy, 400 output tokens. Prompts are WikiText-2 test text cut to length with a
"summarise" instruction; each number is the median of 3 runs. tok/s is the server's own timing.

RTX 3090 24 GB, engine `de8e3c8`, context 40960:

| Prompt | Prefill tok/s | Decode, no drafter | Decode, DFlash2 drafter |
|---|---:|---:|---:|
| 2K | 2,258 | 55.5 | 155.2 |
| 8K | 2,426 | 54.5 | 149.9 |
| 32K | 2,155 | 50.6 | 131.0 |

RTX 5060 Ti 16 GB (WSL 2), engine `be4fd21`, drafter on, one run each: the memory fit chose a 16K context; 2K prompt
777 tok/s prefill / 94.5 tok/s decode, 8K prompt 821 / 76.9.

Speed with the drafter follows how often its guesses are accepted. Summaries of encyclopedia text accept about 35-40 %
of the drafted tokens; reasoning text accepts more, and the LocalMaxxing reasoning prompt runs at 256 tok/s (median of
3) on the RTX 3090 with the same files.

### Reproduce

```sh
sh bench/get_wikitext.sh                       # bench/data/wiki.test.raw (checksum-verified)
# native build: starts tqserve itself, without and with the drafter
python3 bench/relbench.py --bin build/tqserve --gpu "RTX 3090" --wiki bench/data/wiki.test.raw --out results.jsonl \
  --model v2=models/Qwen3.8-27B-TQ3_4S-v2.gguf \
  --drafter models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --vocab models/mtp-vocab/atx_65536.txt
# or against the Docker server above (start it with -e KV_FORMAT=tq3 to match)
python3 bench/relbench.py --url http://localhost:8086 --mode dflash2 --gpu "RTX 3090" \
  --wiki bench/data/wiki.test.raw --out results.jsonl --model v2=models/Qwen3.8-27B-TQ3_4S-v2.gguf
```

Each line of `results.jsonl` records the GPU, the model file, the settings and the engine commit the server reports
(`GET /health`). A second RTX 3090 (also driving a desktop display) reproduced decode within 1 % and prefill 4-15 % lower;
prefill varies more between cards than decode.

## Tokenizer conversion

GGUF models need no tokenizer file: tqserve reads it from the GGUF metadata. Only safetensors checkpoints, or a
deliberate override (`--tokenizer FILE`), need a tokenizer.tqtok. Build it from tokenizer.json and
tokenizer_config.json of the SAME original model family/version. Conversion needs Python 3, not PyTorch:

```sh
python3 tools/tqconvert.py --source /path/to/hf-tokenizer --out-dir models --tokenizer-only
```

## Native build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tqserve -j4
./build/tqserve --model models/model.gguf --port 8086
```

Use a sufficiently recent CUDA toolkit; CUDA 12.9 was tested on RTX 3090.
Model downloads are not included. Model weights and bundled third-party source
retain their own licenses. tqLLM itself is licensed under Apache-2.0 (see LICENSE and NOTICE).

### Request sampling numbers

`top_k` accepts integers from 0 to INT_MAX; sampling currently uses at most 16 candidates.
`seed` accepts integers from 0 to 9007199254740991 (the JSON parser's exact integer range).
Negative seeds are rejected with HTTP 400; omitted, null or zero retains the default seed behavior.
Fractional, out-of-range and nonnumeric values are rejected before integer conversion.

## Target-head shortlist

The default greedy verifier ranks the full vocabulary with a 4-bit head, then re-scores its top 16 candidates with
an int8 head. The re-score is exact for those candidates; an omitted vocabulary row can still win the full int8
projection. This is an empirical speed optimisation, not a certified full-vocabulary argmax. A large shortlist
margin alone does not bound the scores of omitted rows.

For full-vocabulary verification and logit diagnostics, the C++ API provides `Model::set_full_logits(true)`.
Set it before the first forward or verify call on a fresh model: existing CUDA graphs retain the head path captured
when they were created. The reference is the runtime's converted int8 head, not the original GGUF head tensor.
Some wider single-slot verify calls already use that full int8 path; a width sweep must record the path exercised.
