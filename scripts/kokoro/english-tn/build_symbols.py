#!/usr/bin/env python3
# Copyright (c) 2026 Hedy AI
"""Build the second rule_fsts pass: English symbols and standalone uppercase AM/PM.

No model or NeMo dependency. Uses standard tropical OpenFst byte labels.
"""
import argparse
import pathlib
import kaldifst


def build(path):
    boundary = set(b' \t\r\n.,!?;:()[]{}"-')
    symbols = {
        ord(c): (" " + name + " ").encode()
        for c, name in {
            "+": "plus",
            "=": "equals",
            "&": "and",
            "@": "at",
            "#": "hash",
            "*": "times",
            "<": "less than",
            ">": "greater than",
            "%": "percent",
            "$": "dollar",
            "/": "slash",
            "\\": "backslash",
            "_": "underscore",
            "|": "bar",
            "^": "caret",
            "~": "tilde",
        }.items()
    }
    boundary.update(symbols)
    # Pending prefixes are delayed until a boundary disambiguates AM/PM.
    states = {
        name: i
        for i, name in enumerate(
            ("boundary", "inside", "A", "AM", "P", "PM")
        )
    }
    expansions = {"AM": b"a m", "PM": b"p m"}
    arcs = []
    next_state = len(states)

    def add(src, dst, input_label, output):
        nonlocal next_state
        values = list(output) or [0]
        for i, label in enumerate(values):
            target = dst if i == len(values) - 1 else next_state
            if target == next_state:
                next_state += 1
            arcs.append(f"{src} {target} {input_label if i == 0 else 0} {label}")
            src = target

    for state, src in states.items():
        for c in range(1, 256):
            char = bytes([c])
            if state == "boundary" and c in (65, 80):
                add(src, states[chr(c)], c, b"")
            elif state + chr(c) in states and state in ("A", "P"):
                add(src, states[state + chr(c)], c, b"")
            else:
                prefix = state.encode() if state not in ("boundary", "inside") else b""
                if state in expansions and c in boundary:
                    prefix = expansions[state]
                add(
                    src,
                    states["boundary" if c in boundary else "inside"],
                    c,
                    prefix + symbols.get(c, char),
                )
    final = next_state
    next_state += 1
    for state, src in states.items():
        suffix = state.encode() if state not in ("boundary", "inside") else b""
        if state in expansions:
            suffix = expansions[state]
        add(src, final, 0, suffix)
    arcs.append(str(final))
    # OpenFst's text compiler chooses the first source as its start state.
    fst = kaldifst.compile("\n".join(arcs) + "\n", acceptor=False)
    # Symbol names get explicit word boundaries; collapse adjacent whitespace
    # so an infix expression and a spaced expression have identical output.
    collapse_arcs = []
    for state in (0, 1):
        for c in range(1, 256):
            whitespace = c in (9, 10, 13, 32)
            output = (0 if state else 32) if whitespace else c
            collapse_arcs.append(f"{state} {int(whitespace)} {c} {output}")
    collapse_arcs.extend(["0", "1"])
    collapse = kaldifst.compile("\n".join(collapse_arcs) + "\n", acceptor=False)
    kaldifst.arcsort(fst, sort_type="olabel")
    kaldifst.arcsort(collapse, sort_type="ilabel")
    fst = kaldifst.compose(fst, collapse)
    path.parent.mkdir(parents=True, exist_ok=True)
    fst.write(str(path))
    tn = kaldifst.TextNormalizer(str(path))
    for text, expected in [
        ("9:30 AM.", "9:30 a m."),
        ("PM", "p m"),
        ("I am calm.", "I am calm."),
        ("AMBER", "AMBER"),
        ("xAM", "xAM"),
        ("AM PM!", "a m p m!"),
        ("AM-PM", "a m-p m"),
        ("PM-AM", "p m-a m"),
        ("AI in the US.", "AI in the US."),
        ("USA AIM", "USA AIM"),
        ("2+2=4", "2 plus 2 equals 4"),
        ("2 + 2 = 4", "2 plus 2 equals 4"),
    ]:
        assert tn.normalize(text) == expected, (text, tn.normalize(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    build(parser.parse_args().output)
