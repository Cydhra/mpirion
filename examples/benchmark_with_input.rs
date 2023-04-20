use criterion::{BenchmarkId, Criterion};
use mpi::collective::CommunicatorCollectives;
use mpi::traits::Communicator;
use mpirion::{mpirion_bench, mpirion_group, mpirion_kernel, mpirion_main};

fn collective_comm_benchmark(c: &mut Criterion, world: &dyn Communicator) {
    let mut g = c.benchmark_group("collective-comm");
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256].into_iter() {
        g.bench_with_input(BenchmarkId::new("message-size", size), &size, |b, &size| {
            mpirion_bench!(collective_comm_kernel, b, world, size)
        });
    }
    g.finish();
}

fn setup(comm: &dyn Communicator, size: u32) -> Vec<u64> {
    (0u64..comm.size() as u64 * size as u64).collect()
}

fn collective_comm_kernel(comm: &dyn Communicator, data: &mut Vec<u64>) {
    let mut recv_buffer = vec![0u64; data.len()];
    comm.all_to_all_into(data, &mut recv_buffer);
}

mpirion_kernel!(collective_comm_kernel, setup, u32);
mpirion_group!(benches, collective_comm_benchmark);
mpirion_main!(benches, collective_comm_kernel);
