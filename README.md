# Benchmarking MPI algorithms with Criterion
This repository contains a set of macros to set up and run benchmarks for MPI using 
[Criterion](https://github.com/bheisler/criterion.rs).
The benchmarks are supposed to run with [cargo-mpirun](https://github.com/AndrewGaspar/cargo-mpirun) (or just mpiexec).
Unfortunately, this means that the benchmarks cannot be implemented on top of Rust's built-in benchmarking framework,
because those benchmarks are not compiled into a binary with predictable path.
So instead, benchmarks created with these macros are implemented as examples, which are run with cargo-mpirun.
The benchmarks accept all the same command line arguments as Criterion, but you need to pass them manually.

## Usage
To use the macros, add the following to your `Cargo.toml`:
```toml
[dependencies]
mpi = { version = "0.6", git = "https://github.com/Cydhra/rsmpi", branch="sized_process" }

# snip

[dev-dependencies]
criterion = "0.4"
mpirion = { version = "0.1", git="https://github.com/Cydhra/mpirion" }
```

See example benchmarks in `examples/`.
There are two flavors of the macros, one which accepts 
[benchmarks with input](https://bheisler.github.io/criterion.rs/book/user_guide/benchmarking_with_inputs.html) and
one which does not. You can find examples for both.

Currently, the library makes use of a fork of [rsmpi](https://github.com/rsmpi/rsmpi), so you need to use the 
`sized_process` branch of the fork. Those features will hopefully be merged into the main repository soon.

## Features
I mainly created this library for my own projects, so it currently supports only a limited subset of Criterion features.
If you need more, feel free to open an issue or a pull request; I do plan on maintaining this library.