cmake_minimum_required(VERSION 3.16)
project(stripcz LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(STRIPCZ_USE_OPENMP "Thread-parallel assembly and solvers (OpenMP)" ON)
option(STRIPCZ_USE_MPI    "MPI ensemble driver (Weibull realisations / mesh sweeps)" ON)
option(STRIPCZ_NATIVE     "Compile with -march=native" OFF)
option(STRIPCZ_BUILD_TESTS "Build unit tests" ON)

add_library(stripcz_core
  src/config.cpp
  src/mesh.cpp
  src/sparse.cpp
  src/amg.cpp
  src/random_field.cpp
  src/interface_law.cpp
  src/model.cpp
  src/mechanics.cpp
  src/solver.cpp
  src/output.cpp
)
target_include_directories(stripcz_core PUBLIC src)

# build id = sha256 of the solver sources (+ git revision when available), recorded in every manifest
file(GLOB STRIPCZ_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
list(SORT STRIPCZ_SOURCES)
set(_cat "")
foreach(f ${STRIPCZ_SOURCES})
  file(SHA256 ${f} _h)
  string(APPEND _cat ${_h})
endforeach()
string(SHA256 _src_hash "${_cat}")
string(SUBSTRING ${_src_hash} 0 12 _src_hash)
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE _git OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(_git)
  set(STRIPCZ_BUILD_ID "src-${_src_hash}-git-${_git}")
else()
  set(STRIPCZ_BUILD_ID "src-${_src_hash}")
endif()
target_compile_definitions(stripcz_core PRIVATE BSF_BUILD_ID="${STRIPCZ_BUILD_ID}")
target_compile_options(stripcz_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
if(STRIPCZ_NATIVE)
  target_compile_options(stripcz_core PUBLIC -march=native)
endif()

if(STRIPCZ_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(stripcz_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "stripcz: OpenMP enabled")
  else()
    message(WARNING "stripcz: OpenMP requested but not found - building serial")
  endif()
endif()

add_executable(stripcz src/main.cpp)
target_link_libraries(stripcz PRIVATE stripcz_core)

if(STRIPCZ_USE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    target_link_libraries(stripcz PRIVATE MPI::MPI_CXX)
    target_compile_definitions(stripcz PRIVATE STRIPCZ_HAVE_MPI=1)
    message(STATUS "stripcz: MPI ensemble driver enabled")
  else()
    message(STATUS "stripcz: MPI not found - ensemble mode runs cases sequentially")
  endif()
endif()

if(STRIPCZ_BUILD_TESTS)
  enable_testing()
  add_executable(stripcz_tests tests/test_main.cpp)
  target_link_libraries(stripcz_tests PRIVATE stripcz_core)
  add_test(NAME unit COMMAND stripcz_tests)
endif()
