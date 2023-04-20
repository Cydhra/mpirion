use criterion::Criterion;
use mpi::collective::SystemOperation;
use mpi::traits::{Communicator, Root};
use mpirion::{mpirion_bench, mpirion_group, mpirion_kernel, mpirion_main};

fn broadcast_benchmark(c: &mut Criterion, world: &dyn Communicator) {
    c.bench_function("broadcast", |b| mpirion_bench!(broadcast_kernel, b, world));
}

fn reduce_benchmark(c: &mut Criterion, world: &dyn Communicator) {
    c.bench_function("reduce", |b| mpirion_bench!(reduce_kernel, b, world));
}

fn setup(comm: &dyn Communicator) -> u64 {
    comm.rank() as u64
}

fn broadcast_kernel(comm: &dyn Communicator, data: &mut u64) {
    comm.process_at_rank(0).broadcast_into(data);
}

fn reduce_kernel(comm: &dyn Communicator, data: &u64) {
    if comm.rank() == 0 {
        let mut recv_buffer = 0u64;
        comm.process_at_rank(0)
            .reduce_into_root(data, &mut recv_buffer, SystemOperation::sum());
    } else {
        comm.process_at_rank(0)
            .reduce_into(data, SystemOperation::sum());
    }
}

mpirion_kernel!(broadcast_kernel, setup);
mpirion_kernel!(reduce_kernel, setup);
mpirion_group!(broadcast_bench, broadcast_benchmark);
mpirion_group!(reduce_bench, reduce_benchmark);
// named parameters syntax is required to avoid ambiguity when multiple groups are used
mpirion_main!(groups = broadcast_bench, reduce_bench; kernels = broadcast_kernel, reduce_kernel);
