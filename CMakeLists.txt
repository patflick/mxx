cmake_minimum_required(VERSION 3.6)

# project settings
project(mxx LANGUAGES CXX)

#### options
OPTION(MXX_BUILD_TESTS "Build tests" ON)
OPTION(MXX_BUILD_UTILS "Build utilities and benchmarks" OFF)
OPTION(MXX_ENABLE_COVERAGE "Enable code coverage reporting" OFF)
OPTION(MXX_TRAVIS "Travis build (turns on tests and coverage)" OFF)
OPTION(MXX_FAKE_BIG_MPI "Enable a fake, tiny bigMPI threshold (used during CI testing)" OFF)

#### add mxx as header library
add_library(mxx INTERFACE)
target_compile_features(mxx INTERFACE cxx_std_11)
target_include_directories(mxx INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)


#### load MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    target_link_libraries(mxx INTERFACE ${MPI_CXX_LIBRARIES})
    target_include_directories(mxx INTERFACE ${MPI_CXX_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

#### cxx-prettyprint
target_include_directories(mxx INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext>
  $<INSTALL_INTERFACE:include/mxx/ext>)


#### Installation
install(DIRECTORY include/mxx/ DESTINATION include/mxx)
install(DIRECTORY ext/ DESTINATION include/mxx FILES_MATCHING PATTERN "*.hpp")

include(CMakePackageConfigHelpers)

install(TARGETS mxx
        EXPORT mxxTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

export(EXPORT mxxTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/mxx/mxxTargets.cmake"
  NAMESPACE mxx::
)
set(ConfigPackageLocation lib/cmake/mxx)
configure_package_config_file(cmake/mxxConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/mxx/mxxConfig.cmake"
  INSTALL_DESTINATION "${ConfigPackageLocation}"
)

install(EXPORT mxxTargets
  FILE
    mxxTargets.cmake
  NAMESPACE
    mxx::
  DESTINATION
    ${ConfigPackageLocation}
)
install(
  FILES
    cmake/mxxConfig.cmake
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)


#### Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  message(STATUS "Found Doxygen")
  message(STATUS "    use `make doc` to build documentation")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)

#### Travis
if(MXX_TRAVIS)
  # enable coverage
  SET(MXX_BUILD_TESTS ON CACHE BOOL "Build tests on travis" FORCE)
  SET(MXX_ENABLE_COVERAGE ON CACHE BOOL "Enable code coverage reporting" FORCE)
  # set debug build
  SET(CMAKE_BUILD_TYPE Debug)
endif(MXX_TRAVIS)

#### Test Coverage
if(MXX_ENABLE_COVERAGE)
  # turn off stack protection for gcov coverage, because the stack protector shows
  # up as a never taken branch, and thus turns any last statement in a function
  # with a stack procetor into a partially covered statement.
  #
  # additionally: disable inlining for more precise coverage
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-stack-protector -fno-inline -fno-inline-small-functions -fno-default-inline")
endif(MXX_ENABLE_COVERAGE)

#### Fake BIG MPI
# for testing of BigMPI functions on travis or other small RAM build systems
if (MXX_FAKE_BIG_MPI)
  add_definitions(-DMXX_MAX_INT=3)
endif(MXX_FAKE_BIG_MPI)

###### Executable and Libraries
# Save libs and executables in the same place
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# build mxx enabled gtest module
add_subdirectory(gtest)

# build tests
if(MXX_BUILD_TESTS)
add_subdirectory(test)
endif(MXX_BUILD_TESTS)

# build utils and benchmarks
if (MXX_BUILD_UTILS)
add_subdirectory(src)
endif (MXX_BUILD_UTILS)
