/// Generate a main method that configures criterion and initializes MPI, or runs the kernel
/// function if it is called for an MPI spawned process. Only one group can be defined per benchmark,
/// so this macro does not take a variadic list of groups.
///
/// The main method will panic if it is called without any arguments. If the first argument is
/// ``--child``, the kernel function will be executed. Otherwise, the benchmark group will be executed.
///
/// If the benchmark group is called, it accepts all CLI parameters that Criterion usually accepts.
///
/// The macro takes the kernel function as its second argument. The kernel function must take a
/// ``&dyn Communicator`` as its first argument, and a mutable reference to the data type that is
/// returned by the setup function as its second argument. The communicator is the intra-communicator
/// of the spawned child processes. It runs the MPI code that is being benchmarked.
///
/// # Example
/// ```rust
/// use criterion::Criterion;
/// use mpi::traits::Communicator;
/// use mpirion::*;
///
/// fn bench_func(c: &mut Criterion, world: &dyn Communicator) {
///     c.bench_function("my-bench", |b| {
///        mpirion_bench!(world, b)
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
#[macro_export]
macro_rules! mpirion_main {
    ( $group:path, $kernel:path $(,)? ) => {
        fn main() {
            let mut args = std::env::args();

            if let Some(p) = args.nth(1) {
                if p == "--child" {
                    execute_kernel();
                } else {
                    $group();

                    criterion::Criterion::default()
                        .configure_from_args()
                        .final_summary();
                }
            } else {
                panic!("Expected cli arguments for criterion or for MPI child process.")
            }
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
        pub fn $name() {
            let mut criterion: criterion::Criterion<_> = $config
                .configure_from_args();

            let universe = mpi::initialize().unwrap();
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
        fn execute_kernel() {
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
    };
}

/// Generate the communication and spawning code for a benchmark. This macro must be called inside
/// the ``criterion::Criterion::bench_function`` closure. The macro will spawn child processes
/// and then supply the child processes with the number of iterations that should be executed.
///
/// After the child processes have finished, the macro will receive he results of the child processes
/// and calculate the average of them. The average is then used to create a benchmark result.
///
/// # Example
/// See ``mpirion_main!``.
#[macro_export]
macro_rules! mpirion_bench {
    ($world:ident, $bencher:ident $(, $argument:ident)?) => {
        $bencher.iter_custom(|mut iterations| {
            // create child processes
            let mut child_exe = std::process::Command::new(std::env::current_exe().expect("failed to retrieve benchmark executable path"));
            child_exe.arg("--child");

            let child_inter_comm = mpi::collective::Root::spawn(
                &$world.process_at_rank(0),
                &child_exe,
                4, // TODO make this a parameter
            ).expect("failed to spawn child processes");
            let child_world_size = child_inter_comm.remote_size();
            assert_eq!(child_world_size, 4);

            // create intracomm for parent and the children
            let merged_comm = child_inter_comm.merge(mpi::topology::MergeOrder::Low);

            mpi::collective::Root::broadcast_into(&merged_comm.this_process(), &mut iterations);
            $(
                let mut input = $argument.clone();
                merged_comm.this_process().broadcast_into(&mut input);
            )?

            let mut total_nanos: u64 = 0;
            mpi::collective::Root::reduce_into_root(&merged_comm.this_process(), &[0u64], &mut total_nanos, mpi::collective::SystemOperation::sum());
            total_nanos = (total_nanos as f64 / child_world_size as f64) as u64;
            std::time::Duration::from_nanos(total_nanos)
        })
    }
}
