
#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/benchmark.hpp>
#include <mxx/utils.hpp>

#include <ext/cxx-prettyprint/prettyprint.hpp>

std::string exec_name;

void print_usage() {
    std::cerr << "Usage: " << exec_name << " <n> <out-node-filename>" << std::endl;
    std::cerr << "where" << std::endl;
    std::cerr << "    <n>                     Number of nodes to vote off." << std::endl;
    std::cerr << "    <out-node-filename>     Filename for the new nodefile, output by this program." << std::endl;
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
    if (num_nodes % 2 != 0) {
        std::cerr << "Error: this benchmark assumes an even number of nodes" << std::endl;
        MPI_Abort(comm, -1);
    }

    /*
    // parse input arguments
    exec_name = argv[0];
    if (argc < 3) {
        print_usage();
        MPI_Abort(comm, -1);
    }
    int n_vote_off = atoi(argv[1]);
    std::string output_nodefile(argv[2]);
    if (n_vote_off < 0) {
        print_usage();
        MPI_Abort(comm, -1);
    }
    */


    std::vector<double> bw_row = mxx::pairwise_bw_matrix(hc);

    mxx::print_bw_matrix_stats(hc, bw_row);

    mxx::bw_all2all(hc.global, hc.local);

    return 0;
}
