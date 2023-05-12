# Synthetic dataset generator that emulates a trie of height L built over an 
# alphabet of a given size A, in which the first P levels are full and the
# rest are sparsely populated with a density factor D.

import random
import string


def next_string(s, alphabet):
    out = s.rstrip(alphabet[-1])
    if out:
        return (
            out[:-1]
            + alphabet[alphabet.index(out[-1]) + 1]
            + alphabet[0] * (len(s) - len(out))
        )
    return alphabet[0] * (len(s) + 1)


def random_string(alphabet, length):
    return "".join(random.choice(alphabet) for _ in range(length))


def generate(length, full_up_to_length, density, alphabet):
    dataset = set()
    full_prefix = alphabet[-1] * (full_up_to_length - 1)
    full_prefixes = []
    for _ in range(len(alphabet) ** full_up_to_length):
        full_prefix = next_string(full_prefix, alphabet)
        dataset.add(full_prefix + random_string(alphabet, length - full_up_to_length))
        full_prefixes.append(full_prefix)

    n = int(len(alphabet) ** length * density)
    while n > 0:
        random_prefix = random.choice(full_prefixes)
        s = random_prefix + random_string(alphabet, length - full_up_to_length)
        if s not in dataset:
            dataset.add(s)
            n -= 1

    return sorted(dataset)


vary_density = [(12, 6, 0.01, 4), (12, 6, 0.1, 4), (12, 6, 0.5, 4)]
vary_alphabet = [(8, 4, 0.01, 12), (8, 4, 0.01, 14), (8, 4, 0.01, 16)]
vary_length = [(14, 7, 0.01, 4), (15, 7, 0.01, 4), (16, 8, 0.01, 4)]
same_dataset_size = [
    (8, 4, 497599 / 92236816, 14),
    (8, 4, 15497 / 8388608, 16),
    (12, 6, 7351 / 885842380864, 14),
]

for L, P, D, A in vary_density + vary_alphabet + vary_length + same_dataset_size:
    alph = string.ascii_lowercase[:A]
    out = generate(L, P, D, alph)
    filename = "syntheticL{}_P{}_D{}_A{}.txt".format(L, P, D, A)
    with open(filename, "w") as f:
        f.write("\n".join(out))
        print("Wrote {} strings to {}".format(len(out), filename))
