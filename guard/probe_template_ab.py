#!/usr/bin/env python3
"""probe_template_ab.py - does the v22 template change behaviour, statistically?

Compares two running instances (one with --chat-template-file ...v22.jinja, one with the
GGUF's embedded template) on the cases where their rendered prompts actually differ.

    python3 guard/probe_template_ab.py --a http://127.0.0.1:8082 --b http://127.0.0.1:8086 \
        --key KEY --n 8

Cases:
  tools   a request that needs a tool call     <- the templates differ here (v22 has the
                                                 "<IMPORTANT> reasoning/tool placement" block)
  answer  think, then answer a short question  <- measures empty answers and leaks
  plain   NO tools at all                      <- falsification: prompts are identical,
                                                 so this case must show NO difference
"""
import argparse, json, statistics, sys, urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "run_bash", "description": "Run a shell command and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]

CASES = {
    "tools": {"tools": TOOLS, "messages": [
        {"role": "system", "content": "You are a careful assistant. Use the tool when it helps."},
        {"role": "user", "content": "How many files are in /tmp right now? Use the tool."}]},
    "answer": {"tools": TOOLS, "messages": [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Think about it, then answer with one sentence: what is 17 * 23?"}]},
    "plain": {"tools": None, "messages": [
        {"role": "user", "content": "Think about it, then answer with one sentence: what is 17 * 23?"}]},
}


def call(base, key, case, model="x", max_tokens=1500, temperature=0.5):
    body = {"model": model, "messages": CASES[case]["messages"], "max_tokens": max_tokens,
            "temperature": temperature}
    if CASES[case]["tools"]:
        body["tools"] = CASES[case]["tools"]
        body["tool_choice"] = "auto"
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    d = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
    ch = d["choices"][0]; m = ch["message"]
    return {"finish": ch.get("finish_reason"),
            "tool_calls": len(m.get("tool_calls") or []),
            "content": len(m.get("content") or ""),
            "reasoning": len(m.get("reasoning_content") or "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--key", default="llamacpp"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cases", default="tools,answer,plain")
    a = ap.parse_args()
    print(f"  A = {a.a}   B = {a.b}   runs per case = {a.n}\n")
    for case in a.cases.split(","):
        rows = {"A": [], "B": []}
        for side, base in (("A", a.a), ("B", a.b)):
            for _ in range(a.n):
                try:
                    rows[side].append(call(base, a.key, case))
                except Exception as e:
                    rows[side].append({"error": str(e)[:60]})
        def rate(rs, pred):
            ok = sum(1 for r in rs if "error" not in r and pred(r))
            return f"{ok}/{len(rs)}"
        print(f"  case {case!r}")
        for side in ("A", "B"):
            rs = rows[side]
            errs = sum(1 for r in rs if "error" in r)
            print(f"    {side}: empty-answer {rate(rs, lambda r: r['content'] == 0):6} "
                  f"| tool_call {rate(rs, lambda r: r['tool_calls'] > 0):6} "
                  f"| finish=length {rate(rs, lambda r: r['finish'] == 'length'):6} "
                  f"| median reasoning {int(statistics.median([r.get('reasoning', 0) for r in rs if 'error' not in r] or [0])):5}c"
                  + (f" | errors {errs}" if errs else ""))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
