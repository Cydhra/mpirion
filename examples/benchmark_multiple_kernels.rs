use criterion::Criterion;
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::traits::Communicator;
use mpirion::{mpirion_bench, mpirion_group, mpirion_kernel, mpirion_main};

fn simple_benchmark(c: &mut Criterion, world: &dyn Communicator) {
    let mut group = c.benchmark_group("cmp-psum-reduce");
    group.bench_function("prefix-sum", |b| mpirion_bench!(first_kernel, b, world));
    group.bench_function("all-reduce", |b| mpirion_bench!(second_kernel, b, world));
    group.finish();
}

fn setup(comm: &dyn Communicator) -> u64 {
    comm.rank() as u64
}

fn first_kernel(comm: &dyn Communicator, data: &u64) {
    let mut recv_buffer = 0u64;
    comm.scan_into(data, &mut recv_buffer, SystemOperation::sum());
}

fn second_kernel(comm: &dyn Communicator, data: &u64) {
    let mut recv_buffer = 0u64;
    comm.all_reduce_into(data, &mut recv_buffer, SystemOperation::sum());
}

mpirion_kernel!(first_kernel, setup);
mpirion_kernel!(second_kernel, setup);
mpirion_group!(benches, simple_benchmark);
mpirion_main!(benches, first_kernel, second_kernel);
