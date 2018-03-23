#include <string>
#include <sstream>
#include <fstream>

#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/benchmark.hpp>
#include <mxx/utils.hpp>

#include <cxx-prettyprint/prettyprint.hpp>

std::string exec_name;

void print_usage() {
    std::cerr << "Usage: " << exec_name << " -m <msg_size> <output-file>" << std::endl;
    std::cerr << "where" << std::endl;
    std::cerr << "    <output-file>     (optional) Filename for the benchmark results (default: 'all2all_benchmark.csv')." << std::endl;
    std::cerr << " -m <max_mem>         (optional) Maximum message space per node (in GB) (default: 32)." << std::endl;
}

int main(int argc, char* argv[]) {
    mxx::env e(argc, argv);
    mxx::comm comm;

    // print out node and rank distribution
    mxx::print_node_distribution(comm);

    // create shared-mem MPI+MPI hybrid communicator
    mxx::hybrid_comm hc(comm);

    // assert same number processors per node
    int proc_per_node = hc.local.size();
    if (!mxx::all_same(proc_per_node, comm)) {
        std::cerr << "Error: this benchmark assumes the same number of processors per node" << std::endl;
        MPI_Abort(comm, -1);
    }

    // assert we have an even number of nodes
    int num_nodes = hc.num_nodes();
    if (num_nodes > 1 && num_nodes % 2 != 0) {
        std::cerr << "Error: this benchmark assumes an even number of nodes" << std::endl;
        MPI_Abort(comm, -1);
    }

    // default args
    size_t mem_per_node_gb = 32; // setting the max experiment at 32 GB per node
    std::string filename = "all2all_benchmark.csv";

    // parse input arguments
    exec_name = argv[0];
    argv++; argc--;
    if (argc >= 2) {
        std::string x(argv[0]);
        if (x == "-m") {
            mem_per_node_gb = atoi(argv[1]);
            argv+=2; argc-=2;
        }
        if (x != "-m" || mem_per_node_gb > 1024 || mem_per_node_gb == 0) {
            print_usage();
            MPI_Abort(comm, -1);
        }
    }
    if (argc > 0) {
        filename = argv[0];
        argv++; argc--;
    }
    if (argc > 0) {
            print_usage();
            MPI_Abort(comm, -1);
    }

    MXX_ASSERT(mxx::all_same(mem_per_node_gb, comm));

    // benchmark all:
    std::ofstream of;
    if (hc.global.rank() == 0) {
        of.open(filename);
        of << "p,nnodes,q,m,n,min,avg,max" << std::endl;
    }

    // 32 GB/node max?
    size_t mempernode = mem_per_node_gb << 30;

    mxx::forall_p2_nnodes_and_ppn(hc, [&](const mxx::hybrid_comm& hc){
        bm_all2all(hc, of, mempernode);
    });

    return 0;
}
