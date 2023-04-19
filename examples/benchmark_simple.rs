use criterion::Criterion;
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::traits::Communicator;
use mpirion::{mpirion_bench, mpirion_group, mpirion_kernel, mpirion_main};

fn simple_benchmark(c: &mut Criterion, world: &dyn Communicator) {
    c.bench_function("prefix-sum", |b| mpirion_bench!(simple_kernel, b, world));
}

fn setup(comm: &dyn Communicator) -> u64 {
    comm.rank() as u64
}

fn simple_kernel(comm: &dyn Communicator, data: &u64) {
    let mut recv_buffer = 0u64;
    comm.scan_into(data, &mut recv_buffer, SystemOperation::sum());
}

mpirion_kernel!(simple_kernel, setup);
mpirion_group!(benches, simple_benchmark);
mpirion_main!(benches, simple_kernel);
