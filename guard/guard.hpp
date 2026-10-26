// guard/guard.hpp - two rescue rules for tqserve, ported from strata-guard.
//
// Both fit the two-phase machinery tqserve already uses for --reasoning-budget:
//
//   phase 1 stops  ->  decide() says what the reply needed  ->  phase 2 continues
//                      with that text appended to the prompt
//
// CLOSE-THINKING (Strata's reasoning_close_retry): the reply ended with its
// thinking block still open, so the client got no answer.  Close the block and
// let the model answer.
//
// ESCAPE-STOP (strata-guard's control-span rule): the reply ended with an
// unterminated quote, code span, code block or half-typed tag, because the model
// emitted the turn-end token as if it were text.  Put the escaped spelling of
// that token into the prompt and continue, so the turn can finish.
//
// The predicate is ported 1:1 from strata-guard's control_span.inside_span
// (rules 1-4); the escape is the same U+200B-after-first-character convention
// the rest of the codebase already uses for control tokens.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace tqguard {

// U+200B zero-width space, as UTF-8.  Inserted after the leading '<' so the
// marker can never re-tokenise into the reserved id.
inline constexpr std::string_view ZWSP = "\xE2\x80\x8B";

// The closing thinking marker, built by concatenation on purpose: no raw control
// token appears in this source.
inline std::string think_close_marker() { return std::string(1, '<') + "/think" + std::string(1, '>'); }

// The reply ended inside its thinking block: reasoning present, no answer yet.
// This is exactly what a reply that stopped mid-thinking looks like on the client
// side, and it needs no token ids (tqserve splits reasoning on the think id, so a
// spelled-out marker never fakes a close).
inline bool thinking_open(const std::string & reasoning, const std::string & text) {
    return text.empty() && !reasoning.empty();
}

// Does the answer end inside something?  answer = the whole answer; window = the
// characters inspected at the end for the local rules (1-3); scan = the characters
// inspected for the code fence/span rule (4), 0 = the whole turn (strata-guard's
// default; a positive value caps the work for very long outputs).
bool inside_span(std::string_view answer, int window = 32, int scan = 0);

// The marker with a zero-width space after its first byte.
std::string escape_marker(std::string_view marker);

struct Decision {
    enum Kind { None, CloseThinking, EscapeStop } kind = None;
    std::string append;    // text for phase 2's prompt; empty for None
};

class Guard {
public:
    Guard(int max_rescues = 5, int window = 32, int scan = 0)
        : max_rescues_(max_rescues), window_(window), scan_(scan) {}

    // finish: "stop" or "length" (the server's value for the phase that just ended).
    // thinking_open: the reply never closed its thinking block (see thinking_open()).
    // eos_marker: the turn-end token's text, e.g. the im_end token with its brackets.
    // produced: reasoning + answer, as the client received them.
    Decision decide(const std::string & reasoning, const std::string & answer,
                    std::string_view finish, bool thinking_open,
                    std::string_view eos_marker, bool thinking_enabled);

    int rescues() const { return rescues_; }

private:
    int max_rescues_;
    int window_;
    int scan_;
    int rescues_ = 0;
    bool rescued_once_ = false;
    std::size_t produced_after_rescue_ = 0;
};

}  // namespace tqguard
