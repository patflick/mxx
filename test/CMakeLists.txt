cmake_minimum_required(VERSION 2.6)

# project settings
project(mxx-test)

######################
#  MPI Google Test:  #
######################

add_library(mpi_gtest mpi_gtest.cpp ../ext/gtest/gtest-all.cc)
target_link_libraries(mpi_gtest ${EXTRA_LIBS} pthread)

add_executable(test_scat test_scat.cpp)
target_link_libraries(test_scat mpi_gtest)

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
