cmake_minimum_required(VERSION 3.22...4.2)
project(h2b_n300_runtime_diagnostic LANGUAGES CXX)

if(NOT DEFINED H2B_SOURCE_ROOT)
  message(FATAL_ERROR "H2B_SOURCE_ROOT is required")
endif()

add_executable(tt_rqm_h2b_runtime_diagnostic probe.cpp)
target_compile_features(tt_rqm_h2b_runtime_diagnostic PRIVATE cxx_std_20)
target_compile_definitions(tt_rqm_h2b_runtime_diagnostic PRIVATE
  H2B_CANDIDATE_SOURCE="${H2B_SOURCE_ROOT}/experimental/tt_metalium_hamiltonian_evolution/src/hamiltonian_evolution_candidate.cpp"
  TT_RQM_H2B_H2A_READER_PATH="${H2B_SOURCE_ROOT}/experimental/tt_metalium_hamiltonian_evolution/kernels/reader_hamiltonian_lowering_step_major.cpp"
  TT_RQM_H2B_H2A_COMPUTE_PATH="${H2B_SOURCE_ROOT}/experimental/tt_metalium_hamiltonian_lowering_compensated/kernels/compute_hamiltonian_lowering_compensated.cpp"
  TT_RQM_H2B_H2A_WRITER_PATH="${H2B_SOURCE_ROOT}/experimental/tt_metalium_hamiltonian_evolution/kernels/writer_hamiltonian_lowering_step_major.cpp"
  TT_RQM_H2B_H1_READER_PATH="${H2B_SOURCE_ROOT}/experimental/tt_metalium_su2_compose/kernels/su2_fused_reader.cpp"
  TT_RQM_H2B_H1_COMPUTE_PATH="${H2B_SOURCE_ROOT}/experimental/tt_metalium_su2_compose/kernels/su2_fused_compute.cpp"
  TT_RQM_H2B_H1_WRITER_PATH="${H2B_SOURCE_ROOT}/experimental/tt_metalium_su2_compose/kernels/su2_fused_writer.cpp"
)
if(NOT TARGET TT::Metalium)
  find_package(TT-Metalium REQUIRED)
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)
target_link_libraries(tt_rqm_h2b_runtime_diagnostic PUBLIC TT::Metalium PRIVATE MPI::MPI_CXX ${HWLOC_LIBRARY} ${NUMA_LIBRARY})
if(TARGET TT::CommonPCH)
  tt_reuse_precompile_headers(tt_rqm_h2b_runtime_diagnostic TT::CommonPCH)
endif()
