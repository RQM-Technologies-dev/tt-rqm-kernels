cmake_minimum_required(VERSION 3.22...4.2)
project(tt_rqm_metalium_su4q_conformance LANGUAGES CXX)

add_executable(tt_rqm_metalium_su4q_conformance)
target_sources(
    tt_rqm_metalium_su4q_conformance
    PRIVATE src/su4q_conformance_candidate.cpp
)
target_compile_features(tt_rqm_metalium_su4q_conformance PRIVATE cxx_std_20)
target_compile_options(tt_rqm_metalium_su4q_conformance PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(
    tt_rqm_metalium_su4q_conformance
    PRIVATE
        TT_RQM_SU4Q_READER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/reader_stage.cpp"
        TT_RQM_SU4Q_LOCAL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/compute_local.cpp"
        TT_RQM_SU4Q_CARTAN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/compute_cartan.cpp"
        TT_RQM_SU4Q_PHASE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/compute_phase.cpp"
        TT_RQM_SU4Q_WRITER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/writer_state.cpp"
)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)
target_link_libraries(
    tt_rqm_metalium_su4q_conformance
    PUBLIC TT::Metalium
    PRIVATE MPI::MPI_CXX ${HWLOC_LIBRARY} ${NUMA_LIBRARY}
)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(tt_rqm_metalium_su4q_conformance TT::CommonPCH)
endif()
