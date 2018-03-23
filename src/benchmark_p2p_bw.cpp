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
    std::cerr << "    <output-file>     (optional) Filename for the pairwise bandwidth matrix (default: 'p2p_bw.csv')." << std::endl;
    std::cerr << " -m <msg_size>        (optional) Message size for each process pair in kilo bytes. (default: 131072 (128 MB))." << std::endl;
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
    size_t msg_size_kB = 128*1024; // 128 MiB per process default
    std::string filename = "p2p_bw.csv";

    // parse input arguments
    exec_name = argv[0];
    argv++; argc--;
    if (argc >= 2) {
        std::string x(argv[0]);
        if (x == "-m") {
            msg_size_kB = atoi(argv[1]);
            argv+=2; argc-=2;
        }
        if (x != "-m" || msg_size_kB > 4*1024*1024 || msg_size_kB == 0) {
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

    MXX_ASSERT(mxx::all_same(msg_size_kB, comm));

    // perform pairwise bandwidth benchmarking
    std::vector<double> bw_row = mxx::pairwise_bw_matrix(hc, msg_size_kB*1024);

    // print out benchmarking results and save as file
    mxx::print_bw_matrix_stats(hc, bw_row);
    mxx::save_matrix_pernode(hc, filename, bw_row);

    return 0;
}
