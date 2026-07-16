cmake_minimum_required(VERSION 3.22...4.2)
project(tt_rqm_metalium_hamiltonian_lowering_compensated LANGUAGES CXX)

set(BASE "${CMAKE_CURRENT_SOURCE_DIR}/../tt_metalium_hamiltonian_lowering")
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)

function(add_h2a_target target compute_path)
    add_executable(${target})
    target_sources(${target} PRIVATE "${BASE}/src/hamiltonian_lowering_candidate.cpp")
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(
        ${target}
        PRIVATE
            TT_RQM_H2A_READER_PATH="${BASE}/src/kernels/reader_hamiltonian_lowering.cpp"
            TT_RQM_H2A_COMPUTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/${compute_path}"
            TT_RQM_H2A_PRIMITIVE_PROBE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/primitive_probe_hamiltonian_lowering_compensated.cpp"
            TT_RQM_H2A_WRITER_PATH="${BASE}/src/kernels/writer_hamiltonian_lowering.cpp"
    )
    target_link_libraries(
        ${target}
        PUBLIC TT::Metalium
        PRIVATE MPI::MPI_CXX ${HWLOC_LIBRARY} ${NUMA_LIBRARY}
    )
    if(TARGET TT::CommonPCH)
        tt_reuse_precompile_headers(${target} TT::CommonPCH)
    endif()
endfunction()

add_h2a_target(
    tt_rqm_metalium_hamiltonian_lowering_compensated_candidate
    kernels/compute_hamiltonian_lowering_compensated.cpp
)
add_h2a_target(
    tt_rqm_metalium_hamiltonian_lowering_compensated_product_diagnostic
    kernels/compute_hamiltonian_lowering_product_diagnostic.cpp
)
add_h2a_target(
    tt_rqm_metalium_hamiltonian_lowering_compensated_value_diagnostic
    kernels/compute_hamiltonian_lowering_value_diagnostic.cpp
)
add_h2a_target(
    tt_rqm_metalium_hamiltonian_lowering_compensated_trig_diagnostic
    kernels/compute_hamiltonian_lowering_trig_diagnostic.cpp
)
add_h2a_target(
    tt_rqm_metalium_hamiltonian_lowering_original_trig_diagnostic
    kernels/compute_hamiltonian_lowering_original_trig_diagnostic.cpp
)
