# Script to compute some statistics on text files containing sorted strings
# separated by newlines.

import os
import sys

if len(sys.argv) < 2:
    print("Missing input file as argument")
    exit(1)


def lcp_len(a, b):
    i = 0
    for x, y in zip(a, b):
        if x == y:
            i += 1
        else:
            return i
    return i


header = [
    "Input file",
    "Number of strings",
    "Total length",
    "Maximum length",
    "Average length",
    "Maximum LCP",
    "Average LCP",
    "Alphabet size",
    "Alphabet",
]

if len(sys.argv) > 2:
    print(",".join(header))

for input_file in sys.argv[1:]:
    prev = ""
    n = 0
    tot_lcp = 0
    tot_len = 0
    max_len = 0
    max_lcp = 0
    alphabet = set()

    with open(input_file) as fp:
        while True:
            curr = fp.readline().rstrip("\n")
            if not curr:
                break
            if curr < prev:
                print("Error: input file is not sorted")
                exit(1)
            n += 1
            alphabet = alphabet.union(set(curr))
            lcp = lcp_len(prev, curr)
            tot_lcp += lcp
            tot_len += len(curr)
            max_len = max(max_len, len(curr))
            max_lcp = max(max_lcp, lcp)
            prev = curr

    stats = [
        os.path.basename(input_file),
        n,
        tot_len,
        max_len,
        tot_len / n,
        max_lcp,
        tot_lcp / n,
        len(alphabet),
        "".join(sorted(list(alphabet))),
    ]
    if len(sys.argv) == 2:
        for a, b in zip(header, stats):
            print(a.ljust(20), b)
    else:
        print("{},{},{},{},{},{},{},{},{}".format(*stats))
