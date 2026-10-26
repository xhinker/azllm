#include "guard.hpp"

namespace tqguard {

// Whitespace for the local rules: strata-guard's _WHITESPACE.
static bool is_ws(char c) {
    switch (c) {
        case ' ': case '\t': case '\r': case '\n': case '\f': case '\v':
            return true;
        default:
            return false;
    }
}

// A backtick counts as OPENING when the character before it is one of these
// (strata-guard's _OPENERS).  A tick after a letter closes a span, so it must not
// open one - "done`" is a legitimate ending, an opening tick is not.
static bool is_open_tick(char c) {
    switch (c) {
        case ' ': case '\t': case '\n': case '(': case '[': case ':':
        case ',': case '-': case '=': case ';':
            return true;
        default:
            return false;
    }
}

// An open code block or inline span at the end of `text` (strata-guard's
// _open_code_span).  A run of >= 3 backticks is a fence: it opens a block, and a
// matching run closes it.  Inside a block, inline ticks are ignored.  A single
// tick opens an inline span only where is_open_tick says it opens one.
static bool open_code_span(std::string_view text) {
    char fence = 0;          // the fence character while inside a block (0 = none)
    bool inline_open = false;
    std::size_t i = 0, n = text.size();
    while (i < n) {
        const char c = text[i];
        if (c != '`') { ++i; continue; }
        std::size_t j = i;
        while (j < n && text[j] == c) ++j;
        const char prev = (i == 0) ? ' ' : text[i - 1];
        if (j - i >= 3) {                          // a fence run: opens or closes a block
            if (fence == 0) fence = c;
            else if (c == fence) fence = 0;
        } else if (fence == 0) {                   // inline ticks, ignored inside a block
            if (inline_open) inline_open = false;
            else if (is_open_tick(prev)) inline_open = true;
        }
        i = j;
    }
    return fence != 0 || inline_open;
}

bool inside_span(std::string_view answer, int window, int scan) {
    if (window < 0) window = 0;
    const std::string_view tail = answer.size() > static_cast<std::size_t>(window)
                                      ? answer.substr(answer.size() - window) : answer;

    // rule 1: an odd number of double quotes in the tail (local on purpose - global
    // quote parity is "too sticky": one lone quote far back kept the guard firing)
    int quotes = 0;
    for (char c : tail) quotes += (c == '"');
    if (quotes % 2) return true;

    // rule 2: ends on '|', or on an opening backtick (trailing whitespace ignored;
    // '>' is deliberately NOT an opener - "`<|im_end|>`" is a legitimate ending)
    std::size_t n = tail.size();
    while (n > 0 && is_ws(tail[n - 1])) --n;
    if (n > 0) {
        const char last = tail[n - 1];
        if (last == '|') return true;
        if (last == '`' && (n == 1 || is_open_tick(tail[n - 2]))) return true;
    }

    // rule 3: a '<' with no '>' and no whitespace after it (a half-typed tag)
    const auto lt = tail.rfind('<');
    if (lt != std::string_view::npos) {
        bool ok = true;
        for (std::size_t i = lt + 1; i < tail.size(); ++i) {
            const char c = tail[i];
            if (c == '>' || is_ws(c)) { ok = false; break; }
        }
        if (ok) return true;
    }

    // rule 4: an open code block or span anywhere in the turn.  Cannot be local - a
    // fenced block is longer than any tail - and cannot be windowed either (a window
    // that starts inside a block reads its closing fence as an opening one), so it
    // reads the whole turn; scan caps that for very long outputs.
    std::string_view whole = answer;
    if (scan > 0 && whole.size() > static_cast<std::size_t>(scan))
        whole = whole.substr(whole.size() - scan);
    return open_code_span(whole);
}

std::string escape_marker(std::string_view marker) {
    if (marker.empty()) return {};
    return std::string(marker.substr(0, 1)) + std::string(ZWSP) + std::string(marker.substr(1));
}

Decision Guard::decide(const std::string & reasoning, const std::string & answer,
                       std::string_view finish, bool thinking_open,
                       std::string_view eos_marker, bool thinking_enabled) {
    Decision d;
    const bool ended = (finish == "stop" || finish == "length");
    if (!ended) return d;                        // cancelled / error: not our business
    if (rescues_ >= max_rescues_) return d;

    // 1. reasoning_close_retry: the reply ended inside its thinking block
    if (thinking_enabled && thinking_open) {
        ++rescues_;
        d.kind = Decision::CloseThinking;
        d.append = reasoning + "\n" + think_close_marker() + "\n\n";
        return d;
    }

    // 2. control-span: the answer ends in something the model was still writing
    if (answer.empty()) return d;
    if (!inside_span(answer, window_, scan_)) return d;

    // ONE-SHOT PER BURST: the escape is appended to the PROMPT, not the answer, so the
    // answer's length is the "what was produced" measure.  A stop with nothing new in
    // the answer since the escape is the same "I am finished" said twice - let it end
    // instead of appending another copy (which is how one quoted marker used to become
    // five).  produced_after_rescue_ is the answer length at the escape, so an
    // unchanged answer on the next stop is "nothing new".
    const std::size_t produced = reasoning.size() + answer.size();
    if (rescued_once_ && produced == produced_after_rescue_) return d;

    ++rescues_;
    rescued_once_ = true;
    d.kind = Decision::EscapeStop;
    d.append = escape_marker(eos_marker);
    produced_after_rescue_ = produced;
    return d;
}

}  // namespace tqguard
