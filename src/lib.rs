#[macro_export]
macro_rules! mpirion_main {
    ( $group:path, $kernel:path $(,)? ) => {
        fn main() {
            let mut args = env::args();

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

#[macro_export]
macro_rules! mpirion_kernel {
    ($target:path, $setup:path $(, $t:ty)?) => {
        fn execute_kernel() {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();

            let inter_comm = world.parent().expect("child could not retrieve parent comm");
            let merged_comm = inter_comm.merge(mpi::topology::MergeOrder::High);

            let mut iterations = 0u64;
            merged_comm.process_at_rank(0).broadcast_into(&mut iterations);

            $(
                let mut input: $t;
                unsafe {
                    input = std::mem::zeroed();
                    merged_comm.process_at_rank(0).broadcast_into(&mut input);
                }
            )?

            let mut total_duration = std::time::Duration::from_secs(0);
            for _ in 0..iterations {
                let mut data = $setup(&world,
                    $(
                        input as $t
                    )?
                );
                world.barrier();
                let start = std::time::Instant::now();
                $target(&world, &mut data);
                total_duration += start.elapsed();
            }
            let nanos = total_duration.as_nanos() as u64;
            merged_comm.process_at_rank(0).reduce_into(&nanos, SystemOperation::sum());
        }
    };
}

#[macro_export]
macro_rules! mpirion_bench {
    ($world:ident, $iterations:ident $(, $argument:ident)?) => {
        // create child processes
        let mut child_exe = std::process::Command::new(env::current_exe().expect("failed to retrieve benchmark executable path"));
        child_exe.arg("--child");

        let child_inter_comm = $world.process_at_rank(0).spawn(
            &child_exe,
            4, // TODO make this a parameter
        ).expect("failed to spawn child processes");
        let child_world_size = child_inter_comm.remote_size();
        assert_eq!(child_world_size, 4);

        // create intracomm for parent and the children
        let merged_comm = child_inter_comm.merge(mpi::topology::MergeOrder::Low);

        merged_comm.this_process().broadcast_into(&mut $iterations);
        $(
            let mut input = $argument.clone();
            merged_comm.this_process().broadcast_into(&mut input);
        )?

        let mut total_nanos: u64 = 0;
        merged_comm.this_process().reduce_into_root(&[0u64], &mut total_nanos, SystemOperation::sum());
        total_nanos = (total_nanos as f64 / child_world_size as f64) as u64;
        return std::time::Duration::from_nanos(total_nanos);
    }
}
