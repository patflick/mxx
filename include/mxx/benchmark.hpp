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
#include <algorithm>

// MXX
#include "comm.hpp"
#include "datatypes.hpp"
#include "collective.hpp"
#include "timer.hpp"
#include "utils.hpp"

namespace mxx {

void benchmark_nodes_bw_p2p(const mxx::comm& comm = mxx::comm()) {

    size_t n = 10000000;

    std::vector<size_t> vec(n);
    std::generate(vec.begin(), vec.end(), std::rand);

    // pair with another node
    mxx::comm sm_comm = comm.split_shared();
    // communicator with one process per SM node
    bool participate = sm_comm.rank() == 0;
    mxx::comm ct = comm.split(participate);
    participate = participate && ((ct.size() % 2 == 0) || ct.rank() + 1 < ct.size());
    mxx::comm c = ct.split(participate);;

    if (participate) {
        MXX_ASSERT(c.size() % 2 == 0);
        std::string node_name = get_processor_name();
        // execute pairwise benchmark only with one process per node
        // 1) closest neighbor
        int partner = (c.rank() % 2 == 0) ? c.rank() + 1 : c.rank() - 1;
        std::vector<size_t> result;
        c.barrier();
        auto start = std::chrono::steady_clock::now();
        if (c.rank() % 2 == 0) {
            c.send(vec, partner);
            result = c.recv_vec<size_t>(vec.size(), partner);
        } else {
            result = c.recv_vec<size_t>(vec.size(), partner);
            c.send(vec, partner);
        }
        auto end = std::chrono::steady_clock::now();
        double time_p2p = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        c.barrier();
        // calculate BW
        double bw = (double)n*sizeof(size_t)/time_p2p;
        mxx::sync_cout(c) << "[" << node_name << "]: Approx BW= " << bw << " MB/s [" << time_p2p/1000 << "ms" << std::endl;

        // 2) furthest neighbor
    }
    // wait for other processes to finish the benchmarking
    comm.barrier();
}

} // namespace mxx

#endif // MXX_BENCHMARKS_HPP
