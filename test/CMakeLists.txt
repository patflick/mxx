cmake_minimum_required(VERSION 2.6)

# project settings
project(mxx-test)

add_executable(test-collective test_collective.cpp)
target_link_libraries(test-collective mxx-gtest-main)

add_executable(test-reductions test_reductions.cpp)
target_link_libraries(test-reductions mxx-gtest-main)

add_executable(test-send test_send.cpp)
target_link_libraries(test-send mxx-gtest-main)

add_executable(test-sort test_sort.cpp)
target_link_libraries(test-sort mxx-gtest-main)

add_executable(test-distribution test_distribution.cpp)
target_link_libraries(test-distribution mxx-gtest-main)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks mxx-gtest-main)

# TODO: sequential tests should use different main
add_executable(test-bucketing test_bucketing.cpp)
target_link_libraries(test-bucketing mxx-gtest-main)

add_executable(test-all test_collective.cpp test_reductions.cpp test_send.cpp test_sort.cpp test_distribution.cpp)
target_link_libraries(test-all mxx-gtest-main)

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
