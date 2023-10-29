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
mpi = { version = "0.7", git = "https://github.com/Cydhra/rsmpi", branch="allow_dyn_comm" }

# snip

[dev-dependencies]
criterion = "0.5"
mpirion = { version = "0.2", git="https://github.com/Cydhra/mpirion" }
```

The benchmark structure is similar to Criterion's, but you need to use the `mpirion` macros instead of `criterion`.
Additionally, you need one extra method which contains the MPI calls you want to benchmark.

```rust
// This method is only called on the root process and contains the benchmark setup.
// It spawns the child processes which run the kernel and measure the time it takes to 
// run the kernel. The times are then send to the root process and passed to criterion.
fn simple_benchmark(c: &mut Criterion, world: &dyn Communicator) {
    c.bench_function("prefix-sum", |b| mpirion_bench!(simple_kernel, b, world));
}

// This method is called once per iteration (on each MPI process) and returns the data
// that is passed to the kernel, but is not included in the benchmarked time.
fn setup(comm: &dyn Communicator) -> u64 {
    comm.rank() as u64
}

// This method is called once per iteration (on each MPI process) and contains the MPI
// calls you want to benchmark. It is not called on the root process, and the root 
// process will not be included in the communicator.
fn simple_kernel(comm: &dyn Communicator, data: &u64) {
    let mut recv_buffer = 0u64;
    comm.scan_into(data, &mut recv_buffer, SystemOperation::sum());
}

mpirion_kernel!(simple_kernel, setup);
mpirion_group!(benches, simple_benchmark);
mpirion_main!(benches, simple_kernel);
```

See full example benchmarks in `examples/`.
There are two flavors of the macros, one which accepts 
[benchmarks with input](https://bheisler.github.io/criterion.rs/book/user_guide/benchmarking_with_inputs.html) and
one which does not. You can find examples for both.

Currently, the library makes use of a fork of [rsmpi](https://github.com/rsmpi/rsmpi), so you need to use the 
`allow_dyn_comm` branch of the fork. Those features will hopefully be merged into the main repository soon.

## Running the benchmarks
To run the benchmarks, you can use cargo-mpirun.
You can install it with `cargo install cargo-mpirun`.
Then, you can run the benchmarks with `cargo mpirun -n 1 --release --example simple_benchmark -- --bench`.
The `--bench` argument launches the master process, which spawns the child processes and runs the benchmark kernels.
You can pass all normal CLI arguments for Criterion after the `--bench` argument.
If `--bench` is not passed, the executable expects `--child` as first argument,
and will a kernel specified in the second argument within the MPI environment.
Running the benchmarks yourself with `--child` is usually not useful,
it is only used by the master process.
Only one instance of the master process with `--bench` should be run, i.e. you should use `-n 1`.
The amount of child processes is determined by the benchmark itself.

## Missing Features
I mainly created this library for my own projects,
so while designing I only target Criterion features I explicitly need.
For example, the library forces the use of `iter_custom` for measurements,
and currently does not support custom measurement functions that don't accept nanoseconds.
If you need more features,
feel free to open an issue or a pull request.