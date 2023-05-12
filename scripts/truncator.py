# Script that takes a text file containing strings separated by newlines,
# truncates the strings to a given length, deduplicates the result, and writes
# it to a new file.

import sys
import os
import re
import locale
import functools


def remove_prefix(text):
    return re.sub(r"^(?:https?:\/\/)?(?:www\.)?", "", text)


if len(sys.argv) <= 2:
    print("usage: {} <truncate_length> <input_file(s)>".format(sys.argv[0]))
    exit(1)

locale.setlocale(locale.LC_ALL, "C")
truncate = int(sys.argv[1])

for path in sys.argv[2:]:
    with open(path) as f:
        lines = f.read().splitlines()
        n = len(lines)
        lines = [remove_prefix(line) for line in lines]
        lines = [s[:truncate] if len(s) > truncate else s for s in lines]
        lines = sorted(list(set(lines)), key=functools.cmp_to_key(locale.strcoll))
        filename, ext = os.path.splitext(os.path.basename(path))

        out_filename = "{}_truncated{}{}".format(filename, truncate, ext)
        with open(out_filename, "w") as out:
            out.write("\n".join(lines))
            print(f"Wrote {len(lines)} lines to {out_filename} ({n - len(lines)} duplicates removed)")
