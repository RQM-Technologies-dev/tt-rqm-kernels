cmake_minimum_required(VERSION 3.22...4.2)
project(tt_rqm_metalium_qmul_candidate LANGUAGES CXX)

add_executable(tt_rqm_metalium_qmul_candidate)
target_sources(tt_rqm_metalium_qmul_candidate PRIVATE src/qmul_candidate.cpp)
target_compile_features(tt_rqm_metalium_qmul_candidate PRIVATE cxx_std_20)
target_compile_definitions(
    tt_rqm_metalium_qmul_candidate
    PRIVATE
        TT_RQM_QMUL_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/qmul_riscv.cpp"
)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(tt_rqm_metalium_qmul_candidate PUBLIC TT::Metalium)

# The installed TT-Metalium export for the current tt-emule pin does not expose
# all shared-library runtime dependencies needed by libtt_metal.so/UMD.
find_package(MPI REQUIRED COMPONENTS CXX)
find_library(HWLOC_LIBRARY NAMES hwloc REQUIRED)
find_library(NUMA_LIBRARY NAMES numa REQUIRED)
target_link_libraries(
    tt_rqm_metalium_qmul_candidate
    PRIVATE
        MPI::MPI_CXX
        ${HWLOC_LIBRARY}
        ${NUMA_LIBRARY}
)

if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(tt_rqm_metalium_qmul_candidate TT::CommonPCH)
endif()
