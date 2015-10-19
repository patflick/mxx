cmake_minimum_required(VERSION 2.6)

# project settings
project(mxx-gtest)


######################
#  MPI Google Test:  #
######################

include_directories("${PROJECT_SOURCE_DIR}/../")

add_library(mxx-gtest-main mxx_gtest_main.cpp gtest-all.cc)
target_link_libraries(mxx-gtest-main ${MPI_LIBRARIES} pthread)
