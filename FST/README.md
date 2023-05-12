# Fast Succinct Trie

To compile the `benchmark` executable, run the following commands (which require CMake ≥ 3, and a compiler supporting C++17, such as GCC ≥ 7):

```sh
cmake . -DCMAKE_BUILD_TYPE=Release
make -j8
```

The `benchmark` executable takes as arguments the paths to the datasets.
