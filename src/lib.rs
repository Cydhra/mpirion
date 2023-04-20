pub use paste::*;

/// Generate a main method that configures criterion and initializes MPI, or runs a kernel
/// function if it is called for an MPI spawned process. Multiple group targets can be specified,
/// but the user needs to use named parameters syntax to avoid ambiguity with multiple kernel
/// functions.
///
/// The main method will panic if it is called without any arguments. If the first argument is
/// ``--child``, it expects a second argument with the name of the kernel function to execute.
/// Otherwise, the benchmark group will be executed.
///
/// If the benchmark parent is called, it accepts all CLI parameters that Criterion usually accepts.
///
/// The macro takes a variable amount of kernel functions after the group name.
/// Each kernel function must take a ``&dyn Communicator`` as its first argument, and a mutable
/// reference to the data type that is returned by the setup function as its second argument.
/// The communicator is the intra-communicator of the spawned child processes.
/// It runs the MPI code that is being benchmarked.
///
/// # Example
/// ```rust
/// use criterion::Criterion;
/// use mpi::traits::Communicator;
/// use mpirion::*;
///
/// fn bench_func(c: &mut Criterion, world: &dyn Communicator) {
///     c.bench_function("my-bench", |b| {
///        mpirion_bench!(kernel_func, b, world)
///     });
/// }
///
/// fn setup(comm: &dyn Communicator) -> u64 {
///    42
/// }
///
/// fn kernel_func(comm: &dyn Communicator, data: &mut u64) {
///    // do something with data
/// }
///
/// mpirion_kernel!(kernel_func, setup);
/// mpirion_group!(benches, bench_func);
/// mpirion_main!(benches, kernel_func);
/// ```
///
/// Or named parameters syntax when multiple groups are used:
/// ```rust
/// use criterion::Criterion;
/// use mpi::traits::Communicator;
/// use mpirion::{mpirion_bench, mpirion_group, mpirion_kernel, mpirion_main};
///
/// fn broadcast_benchmark(c: &mut Criterion, world: &dyn Communicator) {
///     c.bench_function("kernel1", |b| mpirion_bench!(kernel1, b, world));
/// }
///
/// fn reduce_benchmark(c: &mut Criterion, world: &dyn Communicator) {
///     c.bench_function("kernel2", |b| mpirion_bench!(kernel2, b, world));
/// }
///
/// fn setup(comm: &dyn Communicator) -> u64 { 42 }
/// fn kernel1(comm: &dyn Communicator, data: &mut u64) { /* do something with data */ }
/// fn kernel2(comm: &dyn Communicator, data: &u64) { /* do something with data */ }
///
/// mpirion_kernel!(kernel1, setup);
/// mpirion_kernel!(kernel2, setup);
/// mpirion_group!(kernel1_bench, broadcast_benchmark);
/// mpirion_group!(kernel2_bench, reduce_benchmark);
/// // named parameters syntax is required to avoid ambiguity when multiple groups are used
/// mpirion_main!(groups = kernel1_bench, kernel2_bench; kernels = kernel1, kernel2);
/// ```
#[macro_export]
macro_rules! mpirion_main {
    (groups = $($group:path),+; kernels = $($kernel:path),+) => {
        fn main() {
            let mut args = std::env::args();

            if let Some(p) = args.nth(1) {
                if p == "--child" {
                    if let Some(kernel_arg) = args.next() {
                        match kernel_arg.as_str() {
                            $(
                            stringify!($kernel) => $crate::paste! {[<execute_kernel_ $kernel>]} (),
                            )*
                            _ => panic!("unknown child kernel \"{}\"", kernel_arg),
                        };
                    } else {
                        panic!("called process with --child, but without specifying the kernel");
                    }
                } else {
                    // create universe in main function so MPI is only initialized once
                    let universe = mpi::initialize().unwrap();

                    $(
                    $group(&universe);
                    )*

                    criterion::Criterion::default()
                        .configure_from_args()
                        .final_summary();
                }
            } else {
                panic!("Expected cli arguments for criterion or for MPI child process.")
            }
        }
    };
    ( $group:path, $($kernel:path),+) => {
        $crate::mpirion_main!{
            groups = $group;
            kernels = $($kernel),+
        }
    };
}

/// Generate a criterion benchmark group that initializes MPI for the root process and
/// then calls the target function. This function panics if it isn't run on a single node at rank
/// 0. This macro works the same as criterion's ``criterion_group!``.
/// The child processes are spawned by the benchmark function (assuming it calls ``mpirion_bench!``).
///
/// # Example
/// See ``mpirion_main!``.
#[macro_export]
macro_rules! mpirion_group {
    (name = $name:ident; config = $config:expr; target = $target:path) => {
        pub fn $name(universe: &mpi::environment::Universe) {
            let mut criterion: criterion::Criterion<_> = $config
                .configure_from_args();

            let world = universe.world();
            let rank = world.rank() as usize;
            let world_size = world.size() as usize;

            if rank != 0 {
                panic!("The benchmark root process was run on another node than root. Was run on rank {}.", rank);
            }

            if world_size != 1 {
                eprintln!("The benchmark root process expected to have world size 1, but it has world size {}.", world_size);
            }

            $target(&mut criterion, &world);
        }
    };
    ($name:ident, $target:path $(,)?) => {
        $crate::mpirion_group!{
            name = $name;
            config = criterion::Criterion::default();
            target = $target
        }
    }
}

/// Generate a bootstrap function for MPI child processes. This function will be called by the main
/// function if the first argument is ``--child``. The function will initialize MPI and then
/// repeatedly call the setup function and the kernel function. The results of the kernel function
/// will be sent back to the root process via MPI reduce. This overhead is not included in the
/// benchmark measurements.
///
/// The kernel function must take a ``&dyn Communicator`` as its first argument, and a mutable
/// reference to the data type that is returned by the setup function as its second argument.
/// The communicator is the intra-communicator of the spawned child processes.
///
/// The setup function must take a ``&dyn Communicator`` as its only argument and return the data
/// type that is passed to the kernel function. The setup function is called before each iteration
/// of the kernel function, but it is not included in the benchmark measurements.
///
/// # Example
/// See ``mpirion_main!``.
#[macro_export]
macro_rules! mpirion_kernel {
    ($target:path, $setup:path $(, $t:ty)?) => {
        $crate::paste! {
            fn [<execute_kernel_ $target>] () {
                let universe = mpi::initialize().unwrap();
                let world = universe.world();

                let inter_comm = world.parent().expect("child could not retrieve parent comm");
                let merged_comm = inter_comm.merge(mpi::topology::MergeOrder::High);

                let mut iterations = 0u64;
                mpi::collective::Root::broadcast_into(&merged_comm.process_at_rank(0), &mut iterations);

                $(
                    let mut input: $t;
                    unsafe {
                        input = std::mem::zeroed();
                        mpi::collective::Root::broadcast_into(&merged_comm.process_at_rank(0), &mut input);
                    }
                )?

                let mut total_duration = std::time::Duration::from_secs(0);
                for _ in 0..iterations {
                    let mut data = $setup(&world,
                        $(
                            input as $t
                        )?
                    );
                    mpi::collective::CommunicatorCollectives::barrier(&world);
                    let start = std::time::Instant::now();
                    $target(&world, &mut data);
                    total_duration += start.elapsed();
                }
                let nanos = total_duration.as_nanos() as u64;
                mpi::collective::Root::reduce_into(&merged_comm.process_at_rank(0), &nanos, mpi::collective::SystemOperation::sum());
            }
        }
    };
}

/// Generate the communication and spawning code for a benchmark. This macro must be called inside
/// the ``criterion::Criterion::bench_function`` closure (or one of its variants).
/// The macro will spawn child processes and then supply the child processes with the number of
/// iterations that should be executed.
///
/// After the child processes have finished, the macro will receive he results of the child processes
/// and calculate the average of them. The average is then used to create a benchmark result.
///
/// # Parameters
/// - `kernel` the kernel function that clients run
/// - `bencher` the criterion bencher for collecting results
/// - `world` the current communicator in which the child processes are spawned
/// - `world_size` how many children to spawn. This parameter is optional and defaults to 4.
/// - `argument` optional. An argument to pass to all child processes. This is passed via collective
/// communication. See `examples/benchmark_with_input` for usage: the `mpirion_group!` macro needs
/// to know the argument type, and the kernel setup function needs a parameter for it.
///
/// # Example
/// ```rust
/// use criterion::Criterion;
/// use mpi::traits::Communicator;
/// use mpirion::mpirion_bench;
///
/// // a simple benchmark call without arguments and default world size
/// fn simple_benchmark(c: &mut Criterion, world: &dyn Communicator) {
///     c.bench_function("prefix-sum", |b| mpirion_bench!(simple_kernel, b, world));
/// }
///
/// // if a world_size is required, it can be specified as a named parameter.
/// // Unnamed syntax is not allowed if world_size is required.
/// fn custom_world_benchmark(c: &mut Criterion, world: &dyn Communicator) {
///    c.bench_function("prefix-sum", |b| mpirion_bench!(
///         kernel = simple_kernel,
///         bencher = b,
///         world = world,
///         world_size = 8,
///         arg = 42));
/// }
/// ```
///
/// For a full benchmark example see ``mpirion_main!`` or the ``examples`` directory.
#[macro_export]
macro_rules! mpirion_bench {
    ($kernel:path, $bencher:expr, $world:expr $(, $argument:expr)?) => {
        mpirion_bench!(kernel = $kernel, bencher = $bencher, world = $world, world_size = 4 $(, arg = $argument)?)
    };
    (kernel = $kernel:path, bencher = $bencher:expr, world = $world:expr, world_size = $world_size:expr $(, arg = $argument:expr)?) => {
        $bencher.iter_custom(|mut iterations| {
            // create child processes
            let mut child_exe = std::process::Command::new(std::env::current_exe().expect("failed to retrieve benchmark executable path"));
            child_exe.arg("--child");
            child_exe.arg(stringify!($kernel));

            let child_inter_comm = mpi::collective::Root::spawn(
                &$world.process_at_rank(0),
                &child_exe,
                $world_size,
            ).expect("failed to spawn child processes");
            let child_world_size = child_inter_comm.remote_size();
            assert_eq!(child_world_size, $world_size);

            // create intracomm for parent and the children
            let merged_comm = child_inter_comm.merge(mpi::topology::MergeOrder::Low);

            mpi::collective::Root::broadcast_into(&merged_comm.this_process(), &mut iterations);
            $(
                let mut input = $argument.clone();
                mpi::collective::Root::broadcast_into(&merged_comm.this_process(), &mut input);
            )?

            let mut total_nanos: u64 = 0;
            mpi::collective::Root::reduce_into_root(&merged_comm.this_process(), &[0u64], &mut total_nanos, mpi::collective::SystemOperation::sum());
            total_nanos = (total_nanos as f64 / child_world_size as f64) as u64;
            std::time::Duration::from_nanos(total_nanos)
        })
    }
}
