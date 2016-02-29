/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file    benchmarks.hpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements some MPI bandwidth benchmarks.
 */
#ifndef MXX_BENCHMARKS_HPP
#define MXX_BENCHMARKS_HPP

// MPI include
#include <mpi.h>


// C++ includes
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

#include <cxx-prettyprint/prettyprint.hpp>

// MXX
#include "comm.hpp"
#include "datatypes.hpp"
#include "collective.hpp"
#include "timer.hpp"
#include "utils.hpp"

namespace mxx {


// TODO: take vector/iterators instead of `n`
std::pair<double,double> bw_simplex(const mxx::comm& c, int partner, size_t n) {
    std::vector<size_t> vec(n);
    std::vector<size_t> result(n);
    std::generate(vec.begin(), vec.end(), std::rand);

    c.barrier();
    auto start = std::chrono::steady_clock::now();

    if (c.rank() < partner) {
        c.send(vec, partner);
    } else {
        c.recv_into(&result[0], vec.size(), partner);
    }
    auto end1 = std::chrono::steady_clock::now();
    if (c.rank() < partner) {
        c.recv_into(&result[0], vec.size(), partner);
    } else {
        c.send(vec, partner);
    }
    auto end2 = std::chrono::steady_clock::now();
    c.barrier();
    double time1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start).count();
    double time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count();

    // calculate simplex bandwidth
    double bw1 = 8*(double)n*sizeof(size_t)/time1/1000.0;
    double bw2 = 8*(double)n*sizeof(size_t)/time2/1000.0;
    return std::pair<double,double>(bw1, bw2);
}

template <typename T>
double time_duplex(const mxx::comm& c, int partner, const std::vector<T>& sendvec, std::vector<T>& recvvec) {
    MXX_ASSERT(sendvec.size() == recvvec.size());

    size_t n = sendvec.size();
    c.barrier();
    auto start = std::chrono::steady_clock::now();
    mxx::datatype dt = mxx::get_datatype<size_t>();
    // sendrecv for full duplex
    MPI_Sendrecv(const_cast<T*>(&sendvec[0]), n, dt.type(), partner, 0, &recvvec[0], n, dt.type(), partner, 0, c, MPI_STATUS_IGNORE);
    //c.barrier();
    auto end = std::chrono::steady_clock::now();
    double time_p2p = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    //std::string node_name = get_processor_name();
    //mxx::sync_cout(c) << "[" << node_name << "] time: " << time_p2p << std::endl;

    // calculate duplex bandwidth
    //double bw = 2*8*(double)n*sizeof(size_t)/time_p2p/1000.0;
    return time_p2p;
}

template <typename T>
double bw_duplex_per_node(const mxx::comm& c, int partner, const mxx::comm& smc, const std::vector<T>& sendvec, std::vector<T>& recvvec) {
    size_t n = sendvec.size();
    double time = time_duplex(c, partner, sendvec, recvvec);
    double maxtime_duplex = mxx::allreduce(time, mxx::max<double>(), smc);
    double bw = 2*8*n*sizeof(T)*smc.size()/maxtime_duplex/1000.0;
    return bw; // returns bandwidth in Gb/s
}


void bm(const mxx::comm& c, int partner, const mxx::comm& smc) {

    size_t n = 10000000;
    n /= smc.size();
    size_t MB = (n*sizeof(size_t))/1024/1024;
    std::vector<size_t> vec(n);
    std::vector<size_t> result(n);
    std::generate(vec.begin(), vec.end(), std::rand);

    std::string node_name = get_processor_name();

    double timed = time_duplex(c, partner, vec, result);
    double bw_send, bw_recv;
    std::tie(bw_send, bw_recv) = bw_simplex(c, partner, n);

    // calculate BW
    //double max_bw = mxx::allreduce(bw, mxx::max<double>(), smc);
    //double min_bw = mxx::allreduce(bw, mxx::min<double>(), smc);
    //double sum_bwd = mxx::allreduce(bwd, std::plus<double>(), smc);
    double maxtime_duplex = mxx::allreduce(timed, mxx::max<double>(), smc);
    double sum_bwd = 2*8*n*sizeof(size_t)*smc.size()/maxtime_duplex/1000.0;
    double sum_bw_send = mxx::allreduce(bw_send, std::plus<double>(), smc);
    double sum_bw_recv = mxx::allreduce(bw_recv, std::plus<double>(), smc);
    c.with_subset(smc.rank() == 0, [&](const mxx::comm& subcomm) {
        mxx::sync_cout(subcomm) << "[" << node_name << "]: Node BW Duplex = "
        << sum_bwd << " Gb/s (Simplex " << sum_bw_send << " Gb/s send, " << sum_bw_recv
        << " Gb/s recv) [" << MB*smc.size() << " MiB]" << std::endl;
    });
}

// all-pairwise bandwidth between all nodes
void bw_matrix(const mxx::comm& c, const mxx::comm& smc) {
    MXX_ASSERT(c.size() % smc.size() == 0);
    int num_nodes = c.size() / smc.size();
    size_t n = 10000000;
    std::vector<size_t> vec(n);
    std::vector<size_t> result(n);
    std::generate(vec.begin(), vec.end(), std::rand);
    n /= smc.size();
    // nodes get partnered
    int node_idx = c.rank() / smc.size();
    std::vector<double> bw_row;
    if (smc.rank() == 0)
        bw_row.resize(num_nodes);
    for (int dist = 1; dist < num_nodes; dist <<= 1) {
        if (c.rank() == 0) {
            std::cout << "Benchmarking p2p duplex for dist = " << dist << std::endl;
        }
        if (smc.rank() == 0) {
            //std::cout << "dist " << dist << std::endl;
        }
        int partner_block;
        if ((node_idx / dist) % 2 == 0) {
            // to left block
            partner_block = (node_idx/dist + 1)*dist;
        } else {
            partner_block = (node_idx/dist - 1)*dist;
        }
        int inblock_idx = node_idx % dist;
        for (int i = 0; i < dist; ++i) {
            int partner_node;
            if (partner_block >= node_idx)
                partner_node = partner_block + (inblock_idx + i) % dist;
            else
                partner_node = partner_block + (inblock_idx + (dist - i)) % dist;
            int partner = partner_node*smc.size() + smc.rank();
            // double check partners
            std::vector<int> partners = mxx::allgather(partner, c);
            // check correctness of matching
            for (int i = 0; i < c.size(); ++i) {
                if (partners[partners[i]] != i) {
                    std::cout << "wrong partner for i=" << i << " partner[i]=" << partners[i] << "partner[partner[i]] =" << partners[partners[i]] << std::endl;
                }
            }
            c.with_subset(partner_node < num_nodes, [&](const mxx::comm& subc){
                double bw = bw_duplex_per_node(c, partner, smc, vec, result);
                if (smc.rank() == 0) {
                    bw_row[partner_node] = bw;
                }
            });
        }
    }
    std::string node_name = get_processor_name();
    // print matrix
    c.with_subset(smc.rank() == 0, [&](const mxx::comm& subcomm) {
        mxx::sync_cout(subcomm) << "[" << node_name << "]: " << std::fixed << std::setw(4) << std::setprecision(1) << std::setfill(' ') << bw_row <<  std::endl;

        // calculate avg bw per node
        double sum = 0.0;
        for (int i = 0; i < num_nodes; ++i) {
            if (i != node_idx)
                sum += bw_row[i];
        }
        // output avg bw
        mxx::sync_cout(subcomm) << "[" << node_name << "]: Average BW: " << sum / (num_nodes-1) << std::endl;
        // calc overall average
        double allsum = mxx::allreduce(sum, subcomm);
        if (subcomm.rank() == 0) {
            std::cout << "Overall Average BW: " << allsum / ((num_nodes-1)*(num_nodes-1)) << std::endl;
        }
    });
}

// my own hierarchical all2all using MPI_Sendrecv with all pairs of _nodes_
template <typename T>
void myall2all(const T* send, const T* recv, size_t send_count, const mxx::comm& comm) {
    mxx::datatype dt = mxx::get_datatype<size_t>();
}

void bw_all2all(const mxx::comm& c, const mxx::comm& smc) {
    // message size per target processor
    int m = 16*1024;
    std::vector<size_t> els(m*c.size());
    std::generate(els.begin(), els.end(), std::rand);
    std::vector<size_t> rcv(m*c.size());
    mxx::datatype dt = mxx::get_datatype<size_t>();
    c.barrier();
    auto start = std::chrono::steady_clock::now();
    MPI_Alltoall(&els[0], m, dt.type(), &rcv[0], m, dt.type(), c);
    auto end = std::chrono::steady_clock::now();
    // time in microseconds
    double time_all2all = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double max_time = mxx::allreduce(time_all2all, mxx::max<double>(), c);
    double min_time = mxx::allreduce(time_all2all, mxx::min<double>(), c);
    size_t bits_sendrecv = 2*8*sizeof(size_t)*m*(c.size() - smc.size());
    // bandwidth in Gb/s
    double bw = bits_sendrecv / max_time / 1000.0;
    if (c.rank() == 0) {
        std::cout << "All2all bandwidth: " << bw << " Gb/s [min=" << min_time/1000.0 << " ms, max=" << max_time/1000.0 << " ms, local_size=" << bits_sendrecv/1024/1024 << " MiB]" << std::endl;
    }
}

void benchmark_nodes_bw_p2p(const mxx::comm& comm = mxx::comm()) {
    // pair with another node
    mxx::comm sm_comm = comm.split_shared();
    int proc_per_node = sm_comm.size();

    // assert same number processors per node
    if (!mxx::all_same(proc_per_node, comm)) {
        std::cerr << "Error: this benchmark assumes the same number of processors per node" << std::endl;
        MPI_Abort(comm, -1);
    }

    int num_nodes = comm.size() / proc_per_node;
    // each node gets its index via the ranks of the rank0 procs
    int node_idx = -1;
    comm.with_subset(sm_comm.rank() == 0, [&node_idx](const mxx::comm& subcomm){
        node_idx = subcomm.rank();
    });
    MPI_Bcast(&node_idx, 1, MPI_INT, 0, sm_comm);

    if (num_nodes % 2 != 0) {
        std::cerr << "Error: this benchmark assumes an even number of nodes" << std::endl;
        MPI_Abort(comm, -1);
    }

    //for (int local_p = 1; local_p <= proc_per_node; local_p <<= 1) {
    int local_p = sm_comm.size();
        bool participate =  true; //sm_comm.rank() < local_p;
        mxx::comm c = comm.split(participate, node_idx*local_p + sm_comm.rank());
     //   mxx::comm smc = sm_comm.split(participate);

        if (participate) {
            //bw_matrix(c, sm_comm);
            bw_all2all(c, sm_comm);
            /*
            if (c.rank() == 0) {
                std::cout << "Running with " << local_p << "/" << proc_per_node << " processes per node" << std::endl;
            }
            MXX_ASSERT(c.size() % 2 == 0);
            if (local_p > 1) {
                // intranode BW test
                if (c.rank() == 0)
                    std::cout << "Intranode BW test" << std::endl;
                int partner = (c.rank() % 2 == 0) ? c.rank() + 1 : c.rank() - 1;
                bm(c, partner, smc);
            }
            // 1) closest neighbor
            if (c.rank() == 0)
                std::cout << "Closest Neighbor BW test" << std::endl;
            int partner = (node_idx % 2 == 0) ? c.rank() + local_p : c.rank() - local_p;
            bm(c, partner, smc);

            // 2) furthest neighbor
            if (c.rank() == 0)
                std::cout << "Furthest Neighbor BW test" << std::endl;
            partner = (c.rank() < c.size()/2) ? c.rank() + c.size()/2 : c.rank() - c.size()/2;
            bm(c, partner, smc);
            */
        }
    //}
    // wait for other processes to finish the benchmarking
    comm.barrier();
}

} // namespace mxx

#endif // MXX_BENCHMARKS_HPP
