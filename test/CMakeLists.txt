cmake_minimum_required(VERSION 2.6)

# project settings
project(mxx-test)

######################
#  MPI Google Test:  #
######################

add_library(mpi_gtest mpi_gtest.cpp ../ext/gtest/gtest-all.cc)
target_link_libraries(mpi_gtest ${EXTRA_LIBS} pthread)

add_executable(test-collective test_collective.cpp)
target_link_libraries(test-collective mpi_gtest)

add_executable(test-reductions test_reductions.cpp)
target_link_libraries(test-reductions mpi_gtest)

add_executable(test-send test_send.cpp)
target_link_libraries(test-send mpi_gtest)

add_executable(test-sort test_sort.cpp)
target_link_libraries(test-sort mpi_gtest)

add_executable(test-all test_collective.cpp test_reductions.cpp test_send.cpp test_sort.cpp)
target_link_libraries(test-all mpi_gtest)

################
#  Old tests:  #
################
#add_executable(test_samplesort test_sort.cpp)
#target_link_libraries(test_samplesort ${EXTRA_LIBS})
#
#add_executable(test_file test_file.cpp)
#target_link_libraries(test_file ${EXTRA_LIBS})
#
#add_executable(test_all2all test_bigall2all.cpp)
#target_link_libraries(test_all2all ${EXTRA_LIBS})
#
#add_executable(test_reduce reduce.cpp)
#target_link_libraries(test_reduce ${EXTRA_LIBS})
