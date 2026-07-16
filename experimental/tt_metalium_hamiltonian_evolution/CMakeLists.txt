cmake_minimum_required(VERSION 3.22...4.2)
project(tt_rqm_metalium_hamiltonian_evolution LANGUAGES CXX)

set(H2A_COMPENSATED "${CMAKE_CURRENT_SOURCE_DIR}/../tt_metalium_hamiltonian_lowering_compensated")
set(H1_FUSED "${CMAKE_CURRENT_SOURCE_DIR}/../tt_metalium_su2_compose/kernels")

add_executable(tt_rqm_metalium_hamiltonian_evolution_candidate)
target_sources(
    tt_rqm_metalium_hamiltonian_evolution_candidate
    PRIVATE src/hamiltonian_evolution_candidate.cpp
)
target_compile_features(tt_rqm_metalium_hamiltonian_evolution_candidate PRIVATE cxx_std_20)
target_compile_definitions(
    tt_rqm_metalium_hamiltonian_evolution_candidate
    PRIVATE
        TT_RQM_H2B_H2A_READER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/reader_hamiltonian_lowering_step_major.cpp"
        TT_RQM_H2B_H2A_COMPUTE_PATH="${H2A_COMPENSATED}/kernels/compute_hamiltonian_lowering_compensated.cpp"
        TT_RQM_H2B_H2A_WRITER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/writer_hamiltonian_lowering_step_major.cpp"
        TT_RQM_H2B_H1_READER_PATH="${H1_FUSED}/su2_fused_reader.cpp"
        TT_RQM_H2B_H1_COMPUTE_PATH="${H1_FUSED}/su2_fused_compute.cpp"
        TT_RQM_H2B_H1_WRITER_PATH="${H1_FUSED}/su2_fused_writer.cpp"
)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)
target_link_libraries(
    tt_rqm_metalium_hamiltonian_evolution_candidate
    PUBLIC TT::Metalium
    PRIVATE MPI::MPI_CXX ${HWLOC_LIBRARY} ${NUMA_LIBRARY}
)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(tt_rqm_metalium_hamiltonian_evolution_candidate TT::CommonPCH)
endif()
