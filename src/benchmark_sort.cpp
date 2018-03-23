#include <string>
#include <sstream>
#include <fstream>

#include <mxx/env.hpp>
#include <mxx/comm.hpp>
#include <mxx/benchmark.hpp>
#include <mxx/utils.hpp>

#include <cxx-prettyprint/prettyprint.hpp>

std::string exec_name;

// TODO fix usage
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

    // create output file for benchmark
    std::ofstream of;
    if (hc.global.rank() == 0) {
        of.open("bm_samplesort.csv");
        of << "p,nnodes,q,m,n,min,avg,max" << std::endl;
    }

    size_t mempernode = 16ull << 30;

    // input in growing sizes of 2
    typedef std::tuple<size_t, size_t> T;
    for (size_t npn = 1024; npn <= mempernode/sizeof(T)/2; npn <<= 1) {
        // generate input
        std::vector<T> a(npn);
        srand(comm.rank()* 13 + 5);
        std::generate(a.begin(), a.end(), [](){ return std::make_pair<size_t, size_t>(rand(), rand()); });

    }

    // TODO: sorting benchmark

    return 0;
}
