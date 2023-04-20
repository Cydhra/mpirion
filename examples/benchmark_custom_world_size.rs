use criterion::{black_box, BenchmarkId, Criterion};
use mpi::collective::CommunicatorCollectives;
use mpi::traits::Communicator;
use mpirion::{mpirion_bench, mpirion_group, mpirion_kernel, mpirion_main};

fn simple_benchmark(c: &mut Criterion, world: &dyn Communicator) {
    let mut group = c.benchmark_group("gossiping");
    for size in 2..=8 {
        group.bench_with_input(BenchmarkId::new("all-to-all", size), &size, |b, &size|
                // when altering world size, this syntax needs to be used to avoid ambiguity with
                // input arguments passed to clients
                mpirion_bench! {
                    kernel = simple_kernel,
                    bencher = b,
                    world = world,
                    world_size = size
                });
    }
    group.finish();
}

fn setup(comm: &dyn Communicator) -> Vec<u64> {
    vec![comm.rank() as u64; comm.size() as usize]
}

fn simple_kernel(comm: &dyn Communicator, data: &[u64]) {
    let mut recv_buffer = vec![0u64; comm.size() as usize];
    comm.all_to_all_into(data, &mut recv_buffer);
}

mpirion_kernel!(simple_kernel, setup);
mpirion_group!(benches, simple_benchmark);
mpirion_main!(benches, simple_kernel);
