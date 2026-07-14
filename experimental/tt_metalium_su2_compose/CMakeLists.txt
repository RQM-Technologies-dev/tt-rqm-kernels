cmake_minimum_required(VERSION 3.22...4.2)
project(tt_rqm_metalium_su2_compose_candidate LANGUAGES CXX)

add_executable(tt_rqm_metalium_su2_compose_candidate)
target_sources(tt_rqm_metalium_su2_compose_candidate PRIVATE src/su2_compose_candidate.cpp)
target_compile_features(tt_rqm_metalium_su2_compose_candidate PRIVATE cxx_std_20)
target_compile_definitions(
    tt_rqm_metalium_su2_compose_candidate
    PRIVATE
        TT_RQM_SU2_FUSED_READER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/su2_fused_reader.cpp"
        TT_RQM_SU2_FUSED_COMPUTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/su2_fused_compute.cpp"
        TT_RQM_SU2_FUSED_WRITER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/su2_fused_writer.cpp"
        TT_RQM_SU2_UNFUSED_READER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/su2_unfused_reader.cpp"
        TT_RQM_SU2_UNFUSED_COMPUTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/su2_unfused_compute.cpp"
        TT_RQM_SU2_UNFUSED_WRITER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/su2_unfused_writer.cpp"
)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(tt_rqm_metalium_su2_compose_candidate PUBLIC TT::Metalium)

find_package(MPI REQUIRED COMPONENTS CXX)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)
target_link_libraries(
    tt_rqm_metalium_su2_compose_candidate
    PRIVATE MPI::MPI_CXX ${HWLOC_LIBRARY} ${NUMA_LIBRARY}
)

if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(tt_rqm_metalium_su2_compose_candidate TT::CommonPCH)
endif()
