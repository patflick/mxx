/*
 * Copyright 2016 Georgia Institute of Technology
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
 * @file    benchmark.hpp
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
#include <fstream>
#include <algorithm>

#include <cxx-prettyprint/prettyprint.hpp>

// MXX
#include "comm.hpp"
#include "datatypes.hpp"
#include "collective.hpp"
#include "timer.hpp"
#include "utils.hpp"

namespace mxx {

// a hierarchical communicator wrapper for MPI-MPI hybrid programming
// with 3 communicators: 1 global, 1 per node, 1 for node-masters
class hybrid_comm {
public:
    mxx::comm local;
    mxx::comm local_master;
    mxx::comm global;
    std::string node_name;

private:
    hybrid_comm()
        : local(MPI_COMM_NULL),
          local_master(MPI_COMM_NULL),
          global(MPI_COMM_NULL),
          node_name("") {}

public:
    hybrid_comm(const mxx::comm& c) :
        local(c.split_shared()),
        local_master(c.split(local.rank())),
        global(c.split(true, local_master.rank()*local.size() + local.rank())),
        node_name(mxx::get_processor_name()) {
        MXX_ASSERT(mxx::all_same(local.size()));
    }

    hybrid_comm split_by_node(int color) const {
        // split the processes but assert that each node is only in
        // one process
        MXX_ASSERT(mxx::all_same(color, local));
        hybrid_comm result;
        result.local = local.copy();
        result.global = global.split(color);
        result.local_master = local_master.split(color);
        return result;
    }

    // split the local communicators in the same way on all processes
    // This splits the `local` and `global` communicator and leaves the
    // `local_master` as is, assuming that `color` is identical for all
    // processes in a `local_master`.
    hybrid_comm split_local(int color) const {
        // split the processes but assert that each node is only in
        // one process
        MXX_ASSERT(mxx::all_same(color, local_master));
        hybrid_comm result;
        result.local = local.split(color);
        result.local_master = local_master.copy();
        result.global = global.split(color);
        return result;
    }

    // move constructor moves all members
    hybrid_comm(hybrid_comm&& o) = default;

    // executes only with a subset of nodes
    template <typename Func>
    void with_nodes(bool participate, Func func) const {
        int part = participate;
        MXX_ASSERT(mxx::all_same(part, local));
        if (mxx::all_same(part, global)) {
            if (participate) {
                func(*this);
            }
        } else {
            hybrid_comm hc(split_by_node(participate));
            if (participate) {
                func(hc);
            }
        }
        global.barrier();
    }

    // executes only with `ppn` processes per node
    template <typename Func>
    void with_ppn(int ppn, Func func) const {
        MXX_ASSERT(mxx::all_same(ppn, global));
        MXX_ASSERT(mxx::all_of(ppn <= local.size()));

        int participate = local.rank() < ppn;
        if (mxx::all_same(participate, global)) {
            if (participate) {
                func(*this);
            }
        } else {
            hybrid_comm hc(split_local(participate));
            if (participate) {
                func(hc);
            }
        }
        global.barrier();
    }

    int num_nodes() const {
        return global.size() / local.size();
    }

    int node_rank() const {
        MXX_ASSERT(global.rank() / local.size() == local_master.rank());
        return global.rank() / local.size();
    }

    bool is_local_master() const {
        return local.rank() == 0;
    }

    template <typename Func>
    void with_local_master(Func func) const {
        if (local.rank() == 0)
            func();
        global.barrier();
    }
};

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

/**
 * Collectively times a duplex sendrecv with a given partner rank.
 */
template <typename T>
double time_duplex(const mxx::comm& c, int partner, const std::vector<T>& sendvec, std::vector<T>& recvvec) {
    MXX_ASSERT(sendvec.size() == recvvec.size());
    size_t n = sendvec.size();
    mxx::datatype dt = mxx::get_datatype<size_t>();
    c.barrier();
    auto start = std::chrono::steady_clock::now();
    // sendrecv for full duplex
    MPI_Sendrecv(const_cast<T*>(&sendvec[0]), n, dt.type(), partner, 0, &recvvec[0], n, dt.type(), partner, 0, c, MPI_STATUS_IGNORE);
    auto end = std::chrono::steady_clock::now();
    double time_p2p = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return time_p2p;
}

/**
 * Times duplex sendrecv with a partner process and compute the
 * duplex bandwidth per node.
 */
template <typename T>
double bw_duplex_per_node(const mxx::comm& c, int partner, const mxx::comm& smc, const std::vector<T>& sendvec, std::vector<T>& recvvec) {
    size_t n = sendvec.size();
    double time = time_duplex(c, partner, sendvec, recvvec);
    double maxtime_duplex = mxx::allreduce(time, mxx::max<double>(), smc);
    double bw = 2*8*n*sizeof(T)*smc.size()/maxtime_duplex/1000.0;
    return bw; // returns bandwidth in Gb/s
}





/**
 * @brief Executes the given function f(i) for each pair of processes where
 *        there are `size` processes and this process has rank `rank`.
 *        Processes are paired such that in every iteration
 *        f(i) is called on rank `rank` iff simultanously f(rank) is called on
 *        rank `i`.
 *
 *        The function f(i) is called once for all ranks 0,...,size-1,
 *        excluding i = rank.
 *
 *        If in any iteration, no partner is found, f(-1) is called. This happens
 *        if for example the number of processes `size` is an odd number.
 *
 * @param rank  The rank of this process
 * @param size  The number of total processes
 * @param f     The function to be called for each paired rank.
 */
template <typename F>
void pairwise_func(int rank, int size, F f) {
    // pair up blocks of size 2^i in iteration i
    // for each pair of blocks, pair up all combinations via a linear offset
    for (int dist = 1; dist < size; dist <<= 1) {
        int partner_block;
        if ((rank / dist) % 2 == 0) {
            // to left block
            partner_block = (rank/dist + 1)*dist;
        } else {
            partner_block = (rank/dist - 1)*dist;
        }
        int inblock_idx = rank % dist;
        for (int i = 0; i < dist; ++i) {
            int partner;
            if (partner_block >= rank)
                partner = partner_block + (inblock_idx + i) % dist;
            else
                partner = partner_block + (inblock_idx + (dist - i)) % dist;
            if (partner < size) {
                f(partner);
            } else {
                // this rank doesn't have a partner in this iteration
                f(-1);
            }
        }
    }
}


/**
 * @brief Calls f(c, partner) for each pair of processes and f(-1) if there is
 *        no partner in any given round. (see `pairwise_func(int,int,F)`)
 */
template <typename F>
void pairwise_func(const mxx::comm& c, F f) {
    pairwise_func(c.rank(), c.size(), [&](int partner){ f(c, partner);});
}

// called once for each pair of nodes
// if there are multiple processes per node, the processes are matched
// up by their local ranks
// The given function is called using global ranks as the partner index
// If in any iteration there is no partner node, this calls f(-1)
template <typename F>
void pairwise_nodes_func(const hybrid_comm& hc, F f) {
    MXX_ASSERT(mxx::all_same(hc.local.size()));
    pairwise_func(hc.node_rank(), hc.num_nodes(), [&](int partner_node) {
        if (partner_node >= 0) {
            int partner_rank = partner_node * hc.local.size() + hc.local.rank();
            f(partner_rank);
        } else {
            f(-1);
        }
    });
}

// returns a row of pairwise bw benchmark results on each process where smc.rank() == 0
std::vector<double> pairwise_bw_matrix(const hybrid_comm& hc, size_t msg_size) {
    size_t n = msg_size / 8;
    std::vector<size_t> vec(n);
    std::vector<size_t> result(n);
    std::generate(vec.begin(), vec.end(), std::rand);

    // nodes get partnered
    std::vector<double> bw_row;
    if (hc.is_local_master())
        bw_row.resize(hc.num_nodes());

    pairwise_nodes_func(hc, [&](int partner){
        if (partner >= 0) {
            double bw = bw_duplex_per_node(hc.global, partner, hc.local, vec, result);
            if (hc.is_local_master()) {
                int partner_node = partner / hc.local.size();
                bw_row[partner_node] = bw;
            }
        } else {
            // if this node doesn't participate in the benchmarking, it
            // still needs to call the barrier that is otherwise called
            // inside the time_duplex function
            hc.global.barrier();
        }
    });

    return bw_row;
}


void save_matrix_pernode(const hybrid_comm& hc, const std::string& filename, const std::vector<double>& values) {
    std::ofstream of;
    if (hc.global.rank() == 0) {
        of.open(filename);
    }
    hc.with_local_master([&](){
        std::stringstream ss;
        ss << hc.node_name << ",";
        ss << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < values.size(); ++i) {
            ss << values[i];
            if (i+1 < values.size())
                ss << ",";
        }
        // create sync stream
        mxx::sync_os(hc.local_master, of) << ss.str() << std::endl;
    });
}

void print_bw_matrix_stats(const hybrid_comm& hc, const std::vector<double>& bw_row) {
    // print matrix
    hc.with_local_master([&](){
        // print matrix:
        mxx::sync_cout(hc.local_master) << "[" << hc.node_name << "]: " << std::fixed << std::setw(4) << std::setprecision(1) << std::setfill(' ') << bw_row <<  std::endl;

        // calculate avg bw per node
        // calc min and max
        double sum = 0.0;
        double max = 0.0;
        double min = std::numeric_limits<double>::max();
        int num_nodes = hc.num_nodes();
        int node_idx = hc.node_rank();
        for (int i = 0; i < num_nodes; ++i) {
            if (i != node_idx) {
                sum += bw_row[i];
                if (bw_row[i] > max)
                    max = bw_row[i];
                if (bw_row[i] < min)
                    min = bw_row[i];
            }
        }
        double global_max_bw = mxx::allreduce(max, mxx::max<double>(), hc.local_master);
        // output avg bw
        mxx::sync_cout(hc.local_master) << "[" << hc.node_name << "]: Average BW: " << sum / (num_nodes-1) <<  " Gb/s, Max BW: " << max << " Gb/s, Min BW: " << min << " Gb/s" << std::endl;
        // calc overall average
        double allsum = mxx::allreduce(sum, hc.local_master);
        if (hc.local_master.rank() == 0) {
            std::cout << "Overall Average BW: " << allsum / ((num_nodes-1)*(num_nodes-1)) << " Gb/s, Max: " << global_max_bw << " Gb/s" << std::endl;
        }
        // count how many connections are below 50% of max
        int count_bad = 0;
        int count_terrible = 0; // below 20%
        std::vector<int> bad_nodes;
        std::vector<int> terrible_nodes;
        for (int i = 0; i < num_nodes; ++i) {
            if (i != node_idx) {
                if (bw_row[i] < global_max_bw*0.5) {
                    ++count_bad;
                    bad_nodes.push_back(i);
                }
                if (bw_row[i] < global_max_bw*0.2) {
                    ++count_terrible;
                    terrible_nodes.push_back(i);
                }
            }
        }

        hc.local_master.with_subset(count_bad > 0, [&](const mxx::comm& outcomm) {
             mxx::sync_cout(outcomm) << "[" << hc.node_name << "]: " << count_bad << " bad connections (<50% max): " << bad_nodes << std::endl;
        });
        hc.local_master.with_subset(count_terrible > 0, [&](const mxx::comm& outcomm) {
             mxx::sync_cout(outcomm) << "[" << hc.node_name << "]: " << count_terrible << " terrible connections (<20% max): " << terrible_nodes << std::endl;
        });
    });
}


// all-pairwise bandwidth between all nodes
bool vote_off(const hybrid_comm& hc, int num_vote_off, const std::vector<double>& bw_row) {
    int num_nodes = hc.num_nodes();
    int node_idx = hc.node_rank();
    std::vector<bool> voted_off(num_nodes, false);
    hc.with_local_master([&](){
        // vote off bottlenecks: ie. nodes with lots of close to min connection
        // vote off slowest nodes
        // TODO: each node has `k` votes, vote off nodes with most votes
        //int k = hc.local_master.size()/2;
        double max = *std::max_element(bw_row.begin(), bw_row.end());
        double global_max_bw = mxx::allreduce(max, mxx::max<double>(), hc.local_master);
        for (int i = 0; i < num_vote_off; ++i) {
            // determine minimum of those not yet voted off
            double min = std::numeric_limits<double>::max();
            int off = 0;
            if (!voted_off[node_idx]) {
                for (int j = 0; j < num_nodes; ++j) {
                    if (j != node_idx && !voted_off[j]) {
                        if (bw_row[j] < min) {
                            min = bw_row[j];
                            off = j;
                        }
                    }
                }
            }
            double allmin = mxx::allreduce(min, mxx::min<double>(), hc.local_master);
            // vote only if min is within 0.1*max of allmin
            std::vector<int> vote_off(num_nodes, 0);
            std::vector<double> min_off(num_nodes, std::numeric_limits<double>::max());
            if (!voted_off[node_idx] && min <= allmin+0.1*global_max_bw) {
                vote_off[off] = 1;
                min_off[off] = min;
            }
            std::vector<int> total_votes = mxx::reduce(vote_off, 0, hc.local_master);
            std::vector<double> total_min = mxx::reduce(min_off, 0, mxx::min<double>(), hc.local_master);
            int voted_off_idx;
            if (hc.local_master.rank() == 0) {
                typedef std::tuple<int, int, double> vote_t;
                std::vector<vote_t> votes(num_nodes);
                for (int i = 0; i < num_nodes; ++i) {
                    votes[i] = vote_t(i, total_votes[i], total_min[i]);
                }
                // sort decreasing by number votes
                std::sort(votes.begin(), votes.end(), [](const vote_t& x, const vote_t& y) { return std::get<1>(x) > std::get<1>(y) || (std::get<1>(x) == std::get<1>(y) && std::get<2>(x) < std::get<2>(y));});

                // remove items with 0 votes
                auto it = votes.begin();
                while (std::get<1>(*it) > 0)
                    ++it;
                votes.resize(std::distance(votes.begin(), it));

                // print order
                std::cout << "Votes: " << votes << std::endl;
                std::cout << "Voting off node " << std::get<0>(votes[0]) << " with min " << std::get<2>(votes[0]) << "GiB/s of global min " << allmin << " GiB/s" << std::endl;

                voted_off_idx = std::get<0>(votes[0]);
            }
            MPI_Bcast(&voted_off_idx, 1, MPI_INT, 0, hc.local_master);
            voted_off[voted_off_idx] = true;
            // TODO: run all2all benchmark after each voting
        }
    });
    int votedoff = voted_off[node_idx];
    MPI_Bcast(&votedoff, 1, MPI_INT, 0, hc.local);
    return votedoff == 0;
}

inline mxx::sync_ostream sync_ofstream(const mxx::comm& comm, std::ofstream& os, int root = 0) {
    return comm.rank() == root ? sync_ostream(comm, root, os) : sync_ostream(comm, root);
}

void write_new_nodefile(const hybrid_comm& hc, bool participate, const std::string& filename) {
    hc.with_nodes(participate, [&](const hybrid_comm& subhc) {
        subhc.with_local_master([&](){
            // on rank 0 open filename as stream and write out via sync stream?
            std::ofstream ofile;
            if (subhc.global.rank() == 0) {
                ofile.open(filename);
            }
            mxx::sync_ofstream(subhc.local_master, ofile) << hc.node_name << std::endl;
            ofile.close();
        });
    });
}


// benchmark all2all function that outputs to csv file:
// n (in bytes), p, nnodes, m, ppn, min_time, avg_time, max_time

// m: message size, total data send from one process: m*comm.size()
// n = m*p^2
// returns the time taken by this process for the all2all in microseconds
template <typename T>
double timed_all2all_impl(const mxx::comm& c, size_t m) {
    // create arrays and generate random data
    std::vector<T> els(m*c.size());
    //std::generate(els.begin(), els.end(), [](){ return std::rand() % 255; });
    std::vector<T> rcv(m*c.size());
    mxx::datatype dt = mxx::get_datatype<T>();

    /* run the actual MPI_Alltoall and time it */
    c.barrier();
    auto start = std::chrono::steady_clock::now();
    //MPI_Alltoall(&els[0], m, dt.type(), &rcv[0], m, dt.type(), c);
    mxx::all2all(&els[0], m, &rcv[0], c);
    auto end = std::chrono::steady_clock::now();

    /* return the time taken on this process in microseconds */
    double time_all2all = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return time_all2all;
}

double timed_all2all(const mxx::comm& c, size_t m) {
    if (m >= 8 && m % 8 == 0) {
        return timed_all2all_impl<uint64_t>(c, m/8);
    } else if (m >= 4 && m % 4 == 0) {
        return timed_all2all_impl<uint32_t>(c, m/4);
    } else {
        return timed_all2all_impl<unsigned char>(c, m);
    }
}

void bm_all2all(const mxx::hybrid_comm& hc, std::ostream& os, size_t max_size_per_node) {
    const mxx::comm& c = hc.global;
    assert(sizeof(size_t) == 8);
    if (hc.global.rank() == 0) {
        std::cerr << "bm_all2all with np = " << hc.global.size() << ", ppn = " << hc.local.size() << ", nnodes = " << hc.num_nodes() << std::endl;
    }
    for (size_t m = 1; (m*c.size()*hc.local.size()) <= max_size_per_node; m <<= 1) {
        if (m >= std::numeric_limits<int>::max())
            break;
        size_t np = m*c.size(); // data per process
        size_t n = np*c.size(); // total number of bytes globally

        if (hc.global.rank() == 0) {
            std::cerr << "bm_all2all         m = " << m << ", n/p = " << np << ", n = " << n << ", n/node = " << (m*c.size()*hc.local.size()) << std::endl;
        }

        double time_all2all = timed_all2all(c, m);

        /* get min, max and average */
        double max_time = mxx::allreduce(time_all2all, mxx::max<double>(), c);
        double min_time = mxx::allreduce(time_all2all, mxx::min<double>(), c);
        double avg_time = mxx::allreduce(time_all2all, c) / c.size();

        // p, q, ppn, m, n, time_min, time_avg, time_max
        if (c.rank() == 0) {
            os << c.size() << "," << hc.num_nodes() << "," << hc.local.size() << "," << m << "," << n << "," << min_time << "," << avg_time << "," << max_time << std::endl;
        }
    }
}

#if 0
template <typename F>
void bm_coll_function(const mxx::hybrid_comm& hc, std::ostream& os, F f, size_t max_size_per_node) {
    const mxx::comm& c = hc.global;
    for (size_t m = 1; (m*c.size()*hc.local.size()) <= max_size_per_node; m <<= 1) {
        size_t np = m*c.size(); // max data per process
        size_t n = np*c.size(); // max total number of bytes globally (all2all and allgather)

        // execute the timed function
        double val = f(c, m);

        /* get min, max and average */
        double max_val = mxx::allreduce(val, mxx::max<double>(), comm);
        double min_val = mxx::allreduce(val, mxx::min<double>(), comm);
        double avg_val = mxx::allreduce(val, comm) / comm.size();

        // p, q, ppn, m, n, time_min, time_avg, time_max
        if (c.rank() == 0) {
            os << c.size() << "," << hc.num_nodes() << "," << hc.local.size() << "," << m << "," << n << "," << min_time << "," << avg_time << "," << max_time << std::endl;
        }
    }
}
#endif

// next smaller power of 2
uint32_t flp2 (uint32_t x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}


/**
 * @brief Executes the given function `f(hybrid_comm hc_ppn)` for all powers of
 *        2 between the total ppn downto 1.
 *
 * @param hc    Hybrid communicator which is split for all `ppn` values.
 * @param f     The function called for each `ppn` value.
 */
template <typename F>
void forall_p2_ppn(const mxx::hybrid_comm& hc, F f) {
    int ppn = hc.local.size();
    MXX_ASSERT(mxx::all_same(ppn, hc.global));
    for (int q = ppn; q >= 1; q = flp2(q-1)) {
        // split by ppn
        hc.with_ppn(q, std::forward<F>(f));
    }
}

template <typename F>
void forall_p2_nnodes_and_ppn(const mxx::hybrid_comm& hc, F f) {
    int ppn = hc.local.size();
    MXX_ASSERT(mxx::all_same(ppn, hc.global));

    // in decreasing powers of two
    for (int nn = hc.num_nodes(); nn >= 2; nn = flp2(nn-1)) {
        // split by nodes
        hc.with_nodes(hc.node_rank() < nn, [&](const mxx::hybrid_comm& hcn) {
            for (int q = ppn; q >= 1; q = flp2(q-1)) {
                // split by ppn
                hcn.with_ppn(q, std::forward<F>(f));
            }
        });
    }
}

void bm_all2all_forall_q(const mxx::hybrid_comm& hc, std::ostream& os, size_t max_size_per_node) {
    int ppn = hc.local.size();
    MXX_ASSERT(mxx::all_same(ppn, hc.global));

    for (int q = ppn; q >= 1; q = flp2(q)) {
        hc.with_ppn(q, [&](const mxx::hybrid_comm& hcq) {
            bm_all2all(hcq, os, max_size_per_node);
        });
    }
}

// TODO: this isn't working yet
double ping(const mxx::comm& c, int partner, int rounds = 100) {
    // pairwise ping measurement with this process and process of rank `partner`

    int msg = 0;
    std::chrono::steady_clock::time_point rcv_tp, send_tp;
    MPI_Status st;

    if (c.rank() < partner) {
        // i initiate first ping
        send_tp = std::chrono::steady_clock::now();
        MPI_Send(&msg, 1, MPI_INT, partner, 0, c);
    } else {
        //MPI_Recv(&msg, 1, MPI_INT, partner, MPI_ANY_TAG, c, &st);
    }

    int ping_cnt = 0;
    double ping_sum = 0.;
    for (int i = 0; i <= rounds; ++i) {
            MPI_Recv(&msg, 1, MPI_INT, partner, MPI_ANY_TAG, c, &st);
            rcv_tp = std::chrono::steady_clock::now();
            int t = st.MPI_TAG;
            if (t > 0) {
                    // save time diff
                    double time_diff = std::chrono::duration_cast<std::chrono::microseconds>(rcv_tp - send_tp).count();
                    ping_cnt += 1;
                    ping_sum += time_diff;
            }
            if (t < 2*rounds) {
                    send_tp = std::chrono::steady_clock::now();
                    MPI_Send(&msg, 1, MPI_INT, partner, t+1, c);
            }
            if (t+1 == 2*rounds)
                    break;
    }
    assert(ping_cnt == rounds);
    return ping_sum / ping_cnt;
}



void bw_all2all(const mxx::comm& c, const mxx::comm& smc) {
    // message size per target processor
    for (int k = 1; k <= 16; k <<= 1) {
        size_t n = k*1024*1024;
        int m = n/c.size();
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
        size_t bits_sendrecv = 2*8*sizeof(size_t)*smc.size()*m*(c.size() - smc.size());
        // bandwidth in Gb/s
        double bw = bits_sendrecv / max_time / 1000.0;
        if (c.rank() == 0) {
            std::cout << "All2all bandwidth (per node) " << bw << " Gb/s [min=" << min_time/1000.0 << " ms, max=" << max_time/1000.0 << " ms, local_size=" << bits_sendrecv/1024/1024 << " MiB]" << std::endl;
        }
    }
}

void bw_all2all_char(const mxx::comm& c, const mxx::comm& smc) {
    // message size per target processor
    for (int k = 8; k <= 64; k <<= 1) {
        int m = k*1024;
        std::srand(13*c.rank());
        std::vector<size_t> send_counts(c.size());
        for (int i = 0; i < c.size(); ++i) {
            send_counts[i] = m;
        }
        size_t n = std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0));
        std::vector<char> els(n);
        std::generate(els.begin(), els.end(), std::rand);
        std::vector<size_t> recv_counts = mxx::all2all(send_counts, c);
        size_t recv_n = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
        std::vector<char> rcv(recv_n);
        mxx::datatype dt = mxx::get_datatype<char>();
        c.barrier();
        auto start = std::chrono::steady_clock::now();
        mxx::all2allv(&els[0], send_counts, &rcv[0], recv_counts, c);
        //MPI_Alltoall(&els[0], m, dt.type(), &rcv[0], m, dt.type(), c);
        auto end = std::chrono::steady_clock::now();
        // time in microseconds
        double time_all2all = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double max_time = mxx::allreduce(time_all2all, mxx::max<double>(), c);
        double min_time = mxx::allreduce(time_all2all, mxx::min<double>(), c);
        size_t bits_sendrecv = 2*8*sizeof(char)*m*(c.size() - smc.size());
        // bandwidth in Gb/s
        double bw = bits_sendrecv / max_time / 1000.0;
        if (c.rank() == 0) {
            std::cout << "All2all char bandwidth: " << bw << " Gb/s [min=" << min_time/1000.0 << " ms, max=" << max_time/1000.0 << " ms, local_size=" << bits_sendrecv/1024/1024 << " MiB]" << std::endl;
        }
    }
}

void bw_all2all_unaligned_char(const mxx::comm& c, const mxx::comm& smc, bool realign) {
    // message size per target processor
    for (int k = 1; k <= 128; k <<= 1) {
        int m = k*1024;
        std::srand(13*c.rank());

        // send counts
        std::vector<size_t> send_counts(c.size());
        for (int i = 0; i < c.size(); ++i) {
            send_counts[i] = m + std::rand() % 8;
        }
        size_t n = std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0));
        // generate input
        std::vector<char> els(n);
        std::generate(els.begin(), els.end(), std::rand);
        // original recv counts
        std::vector<size_t> recv_counts = mxx::all2all(send_counts, c);
        size_t recv_n = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
        std::vector<char> rcv(recv_n);

        // time all2all
        c.barrier();
        auto start = std::chrono::steady_clock::now();
        /*
        if (realign) {
            char_all2allv(&els[0], send_counts, &rcv[0], recv_counts, c);
        } else {
        */
            mxx::all2allv(&els[0], send_counts, &rcv[0], recv_counts, c);
        //}
        //mxx::datatype dt = mxx::get_datatype<char>();
        //MPI_Alltoall(&els[0], m, dt.type(), &rcv[0], m, dt.type(), c);
        auto end = std::chrono::steady_clock::now();
        if (realign) {
            std::vector<char> rcv2(recv_n);
            mxx::all2allv(&els[0], send_counts, &rcv2[0], recv_counts, c);
            if (!(rcv == rcv2)) {
                std::cout << "[ERROR] Vectors are not same" << std::endl;
                std::cout << "rcv=" << rcv << std::endl;
                std::cout << "rcv2=" << rcv2 << std::endl;
            }
        }
        // time in microseconds
        double time_all2all = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double max_time = mxx::allreduce(time_all2all, mxx::max<double>(), c);
        double min_time = mxx::allreduce(time_all2all, mxx::min<double>(), c);
        size_t bits_sendrecv = 2*8*sizeof(char)*m*(c.size() - smc.size());
        // bandwidth in Gb/s
        double bw = bits_sendrecv / max_time / 1000.0;
        if (c.rank() == 0) {
            std::cout << "All2all UNaligned char bandwidth: " << bw << " Gb/s [min=" << min_time/1000.0 << " ms, max=" << max_time/1000.0 << " ms, local_size=" << bits_sendrecv/1024/1024 << " MiB]" << std::endl;
        }
    }
}


} // namespace mxx

#endif // MXX_BENCHMARKS_HPP
