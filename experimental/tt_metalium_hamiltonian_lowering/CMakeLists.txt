cmake_minimum_required(VERSION 3.22...4.2)
project(tt_rqm_metalium_hamiltonian_lowering_candidate LANGUAGES CXX)

add_executable(tt_rqm_metalium_hamiltonian_lowering_candidate)
target_sources(
    tt_rqm_metalium_hamiltonian_lowering_candidate
    PRIVATE src/hamiltonian_lowering_candidate.cpp
)
target_compile_features(tt_rqm_metalium_hamiltonian_lowering_candidate PRIVATE cxx_std_20)
target_compile_definitions(
    tt_rqm_metalium_hamiltonian_lowering_candidate
    PRIVATE
        TT_RQM_H2A_READER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/reader_hamiltonian_lowering.cpp"
        TT_RQM_H2A_COMPUTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/compute_hamiltonian_lowering.cpp"
        TT_RQM_H2A_PRIMITIVE_PROBE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/primitive_probe_hamiltonian_lowering.cpp"
        TT_RQM_H2A_WRITER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/writer_hamiltonian_lowering.cpp"
)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(tt_rqm_metalium_hamiltonian_lowering_candidate PUBLIC TT::Metalium)

find_package(MPI REQUIRED COMPONENTS CXX)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)
target_link_libraries(
    tt_rqm_metalium_hamiltonian_lowering_candidate
    PRIVATE MPI::MPI_CXX ${HWLOC_LIBRARY} ${NUMA_LIBRARY}
)

if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(tt_rqm_metalium_hamiltonian_lowering_candidate TT::CommonPCH)
endif()
