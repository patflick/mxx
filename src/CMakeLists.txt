cmake_minimum_required(VERSION 2.6)

# project settings
project(mxx-bm)

add_executable(mxx-bm-vote-off vote_off.cpp)
target_link_libraries(mxx-bm-vote-off ${MPI_LIBRARIES})
