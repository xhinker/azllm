// Offline tests: no GPU, no model.
//   g++ -std=c++17 -O2 -Wall -Wextra test_guard.cpp guard.cpp -o /tmp/tg && /tmp/tg
#include <cstdio>
#include <string>
#include "guard.hpp"

using namespace tqguard;
static int failures = 0;
static void check(bool ok, const char *what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const std::string F3(3, '`'), F4(4, '`'), TC = think_close_marker();
    const std::string EOS = std::string(1, '<') + "|im_end" + std::string(1, '>');

    std::printf("predicate (local rules 1-3):\n");
    check(inside_span("say \"hello"), "odd quote");
    check(inside_span("the operator `"), "opening backtick");
    check(inside_span("stop: <|im"), "half-typed tag");
    check(!inside_span("done`"), "closing backtick is fine");
    check(!inside_span("Hello!"), "plain prose");
    check(inside_span("a | "), "trailing pipe, then whitespace");
    check(inside_span("a ` \n"), "opening backtick, then whitespace");
    check(!inside_span("a word`"), "tick after a letter closes (no rescue)");
    check(!inside_span("`<|im_end|>`"), "quoted marker in ticks is a legitimate ending");

    std::printf("\npredicate (rule 4, code fence/span over the whole turn):\n");
    check(inside_span(F3 + "python\nprint(1)"), "open fence");
    check(!inside_span(F3 + "a\n" + F3), "closed fence");
    check(inside_span(F4 + "rust\nfn main(){}"), "open 4-backtick fence");
    check(!inside_span(F4 + "rust\nfn main(){}\n" + F4), "closed 4-backtick fence");
    check(inside_span("use `x` and `y"), "open inline span");
    check(!inside_span("use `x` and `y`"), "paired inline spans");
    check(!inside_span(F3 + "a\n" + F3 + " then `b`"), "closed fence, open inline after");

    std::printf("\nscan knob (0 = whole turn, N = cap):\n");
    {
        // an open fence buried earlier than `scan` chars back is invisible when capped
        const std::string open = F3 + "x\n" + std::string(80, 'a') + " done";
        check(inside_span(open, 32, 0), "scan=0 sees the open fence (whole turn)");
        check(!inside_span(open, 32, 20), "scan=20 misses it (capped)");
    }

    std::printf("\nescape:\n");
    const std::string esc = escape_marker(EOS);
    check(esc.size() == EOS.size() + ZWSP.size(), "escape adds exactly one ZWSP");
    check(esc.find(ZWSP) == 1, "ZWSP sits after the leading '<'");

    std::printf("\nthinking_open helper:\n");
    check(thinking_open("some reasoning", ""), "reasoning present, no answer -> open");
    check(!thinking_open("some reasoning", "the answer"), "answer present -> closed");
    check(!thinking_open("", ""), "nothing at all -> not an open thinking block");

    std::printf("\nreasoning_close_retry (CloseThinking):\n");
    {
        Guard g;
        auto d = g.decide("thinking, no close", "", "stop", true, EOS, true);
        check(d.kind == Decision::CloseThinking, "ends inside thinking -> close and continue");
        check(d.append.find("thinking, no close") == 0, "continuation keeps the reasoning");
        check(d.append.find(TC) != std::string::npos, "continuation carries the close marker");
        auto again = Guard().decide("x", "answer", "stop", false, EOS, true);
        check(again.kind == Decision::None, "closed thinking, clean answer -> nothing");
        auto nothink = Guard().decide("x", "", "stop", true, EOS, false);
        check(nothink.kind == Decision::None, "thinking disabled -> no close-thinking rescue");
    }

    std::printf("\ncontrol-span (EscapeStop):\n");
    {
        Guard g;
        auto d = g.decide("", std::string(1, '`'), "stop", false, EOS, true);
        check(d.kind == Decision::EscapeStop, "answer ends on a backtick -> escape");
        // one-shot (server semantics): the escape goes in the prompt, so the answer is
        // unchanged on the next stop -> nothing new -> end the turn
        auto d2 = g.decide("", std::string(1, '`'), "stop", false, EOS, true);
        check(d2.kind == Decision::None, "one-shot: nothing new since the escape -> end");
        // new output since the escape -> rescue again
        Guard g2;
        g2.decide("", std::string(1, '`'), "stop", false, EOS, true);
        auto d3 = g2.decide("", std::string(1, '`') + " more `", "stop", false, EOS, true);
        check(d3.kind == Decision::EscapeStop, "new output since the escape -> rescue again");
    }

    std::printf("\nlimits:\n");
    {
        Guard g(1);
        check(g.decide("", std::string(1, '`'), "stop", false, EOS, true).kind == Decision::EscapeStop, "first rescue");
        check(g.decide("", "plain text", "stop", false, EOS, true).kind == Decision::None, "cap 1 respected");
        check(g.decide("", "a\"", "cancelled", false, EOS, true).kind == Decision::None, "cancelled is not a stop");
        check(g.rescues() == 1, "rescue counter tracks the one rescue");
    }

    std::printf("\n%s\n", failures ? "SOME FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
