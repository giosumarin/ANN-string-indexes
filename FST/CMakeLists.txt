cmake_minimum_required(VERSION 3.0)
project(FST)

set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -march=native")

include_directories(fast_succinct_trie/include)

#find_package(OpenMP)
#if (OpenMP_CXX_FOUND)
#  link_libraries(OpenMP::OpenMP_CXX)
#endif ()

add_executable(benchmark benchmark.cpp)
