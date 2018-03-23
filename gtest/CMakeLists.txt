cmake_minimum_required(VERSION 3.6)

# project settings
project(mxx-gtest)


######################
#  MPI Google Test:  #
######################

add_library(mxx-gtest-main mxx_gtest_main.cpp gtest-all.cc)
target_link_libraries(mxx-gtest-main PUBLIC mxx ${MPI_LIBRARIES} pthread)
target_include_directories(mxx-gtest-main PUBLIC "../")
