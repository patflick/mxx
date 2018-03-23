cmake_minimum_required(VERSION 3.6)

# project settings
project(mxx LANGUAGES CXX)

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
    set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

add_library(mxx INTERFACE)
target_compile_features(mxx INTERFACE cxx_std_11)
target_include_directories(mxx INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)


#### MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
    #set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    target_link_libraries(mxx INTERFACE ${MPI_CXX_LIBRARIES})
    target_include_directories(mxx INTERFACE ${MPI_CXX_INCLUDE_DIRS})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

#### cxx-prettyprint
target_include_directories(mxx INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext> $<INSTALL_INTERFACE:include/mxx/ext>)

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

### Test Coverage
OPTION(ENABLE_COVERAGE "Enable code coverage reporting" OFF)

# on travis: debug build with coverage
OPTION(TRAVIS "Travis build" OFF)
if(TRAVIS)
  # enable coverage
  SET(ENABLE_COVERAGE ON CACHE BOOL "Enable code coverage reporting" FORCE)
  # set debug build
  SET(CMAKE_BUILD_TYPE Debug)
endif(TRAVIS)

if(ENABLE_COVERAGE)
  # turn off stack protection for gcov coverage, because the stack protector shows
  # up as a never taken branch, and thus turns any last statement in a function
  # with a stack procetor into a partially covered statement.
  #
  # additionally: disable inlining for more precise coverage
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-stack-protector -fno-inline -fno-inline-small-functions -fno-default-inline")
endif(ENABLE_COVERAGE)

### Fake BIG MPI for testing of BigMPI functions on travis or other small RAM
### build systems
OPTION(FAKE_BIG_MPI "Enable a fake bigMPI threshold of 1" OFF)
if (FAKE_BIG_MPI)
  add_definitions(-DMXX_MAX_INT=3)
endif(FAKE_BIG_MPI)

###### Executable and Libraries
# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

# build tests
add_subdirectory(gtest)
add_subdirectory(test)
if (NOT TRAVIS)
add_subdirectory(src)
endif (NOT TRAVIS)
