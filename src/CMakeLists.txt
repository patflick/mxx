cmake_minimum_required(VERSION 2.8)

# project settings
project(mxx-bm)

add_executable(mxx-bm-vote-off vote_off.cpp)
target_link_libraries(mxx-bm-vote-off ${MPI_LIBRARIES})

# benchmark p2p bandwidth
add_executable(mxx-benchmark-p2p-bw benchmark_p2p_bw.cpp)
target_link_libraries(mxx-benchmark-p2p-bw ${MPI_LIBRARIES})

# benchmark all2all
add_executable(mxx-benchmark-a2a benchmark_a2a.cpp)
target_link_libraries(mxx-benchmark-a2a ${MPI_LIBRARIES})

# benchmark parallel sorting
add_executable(mxx-sort-benchmark benchmark_sort.cpp)
target_compile_definitions(mxx-sort-benchmark PUBLIC -DMXX_SAMPLESORT_TIMER=1)
target_link_libraries(mxx-sort-benchmark ${MPI_LIBRARIES})
