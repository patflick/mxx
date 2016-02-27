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

void bm(const mxx::comm& c, int partner, const mxx::comm& smc) {
    size_t n = 10000000;
	n /= smc.size();

    std::vector<size_t> vec(n);
    std::generate(vec.begin(), vec.end(), std::rand);
        std::string node_name = get_processor_name();
        // execute pairwise benchmark only with one process per node
        // 1) closest neighbor
        std::vector<size_t> result(n);
        c.barrier();
        auto start = std::chrono::steady_clock::now();

        if (c.rank() < partner) {
            c.send(vec, partner);
            c.recv_into(&result[0], vec.size(), partner);
        } else {
            c.recv_into(&result[0], vec.size(), partner);
            c.send(vec, partner);
        }

/*
	mxx::datatype dt = mxx::get_datatype<size_t>();
	//MPI_Sendrecv(&vec[0], n, dt.type(), partner, 0, &result[0], n, dt.type(), partner, 0, c, MPI_STATUS_IGNORE);
	MPI_Request req;
	MPI_Isend(&vec[0], n, dt.type(), partner, 0, c, &req);
	MPI_Recv(&result[0], n, dt.type(), partner, 0, c, MPI_STATUS_IGNORE);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
*/
        auto end = std::chrono::steady_clock::now();
        double time_p2p = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        c.barrier();
        // calculate BW
       	size_t MB = 2*(n/1024)*sizeof(size_t)/1024;
        double bw = 2*8*(double)n*sizeof(size_t)/time_p2p/1000.0;
	//double max_bw = mxx::allreduce(bw, mxx::max<double>(), smc);
	//double min_bw = mxx::allreduce(bw, mxx::min<double>(), smc);
	double sum_bw = mxx::allreduce(bw, std::plus<double>(), smc);
	c.with_subset(smc.rank() == 0, [&](const mxx::comm& subcomm) {
		mxx::sync_cout(subcomm) << "[" << node_name << "]: Node BW = " << sum_bw << " Gb/s [" << time_p2p/1000 << "ms, " << MB*smc.size() << " MiB]" << std::endl;
	});
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
	

	for (int local_p = 1; local_p <= proc_per_node; local_p <<= 1) {
	    bool participate = sm_comm.rank() < local_p;
	    mxx::comm c = comm.split(participate, node_idx*local_p + sm_comm.rank());
	    mxx::comm smc = sm_comm.split(participate);

    if (participate) {
	if (c.rank() == 0) {
		std::cout << "Running with " << local_p << "/" << proc_per_node << " processes per node" << std::endl;
	}
        MXX_ASSERT(c.size() % 2 == 0);
	if (local_p > 1) {
	// intranode BW test
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
    }
}
    // wait for other processes to finish the benchmarking
    comm.barrier();
}

} // namespace mxx

#endif // MXX_BENCHMARKS_HPP
