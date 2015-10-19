cmake_minimum_required(VERSION 2.6)

# project settings
project(mxx-gtest)


######################
#  MPI Google Test:  #
######################

add_library(mxx-gtest-main mpi_gtest.cpp ../ext/gtest/gtest-all.cc)
target_link_libraries(mxx-gtest-main ${MPI_LIBRARIES} pthread)
