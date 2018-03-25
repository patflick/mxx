cmake_minimum_required(VERSION 3.6)

# project settings
project(mxx-test)

if (TRAVIS)

# on travis: build a single executable
add_executable(mxx-test-all test_all.cpp)
target_link_libraries(mxx-test-all mxx-gtest-main)

else (TRAVIS)

add_executable(mxx-test-collective test_collective.cpp)
target_link_libraries(mxx-test-collective mxx-gtest-main)

add_executable(mxx-test-reductions test_reductions.cpp)
target_link_libraries(mxx-test-reductions mxx-gtest-main)

add_executable(mxx-test-send test_send.cpp)
target_link_libraries(mxx-test-send mxx-gtest-main)

add_executable(mxx-test-sort test_sort.cpp)
target_link_libraries(mxx-test-sort mxx-gtest-main)

add_executable(mxx-test-distribution test_distribution.cpp)
target_link_libraries(mxx-test-distribution mxx-gtest-main)

# Combination of all parallel tests:
add_executable(mxx-test-all test_collective.cpp test_reductions.cpp test_send.cpp test_sort.cpp test_distribution.cpp)
target_link_libraries(mxx-test-all mxx-gtest-main)

# TODO: sequential tests should use different main
# TODO: these should also be tested on travis!
add_executable(mxx-test-bucketing test_bucketing.cpp)
target_link_libraries(mxx-test-bucketing mxx-gtest-main)

add_executable(mxx-test-typetraits test_typetraits.cpp)
target_link_libraries(mxx-test-typetraits mxx-gtest-main)

endif (TRAVIS)



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
