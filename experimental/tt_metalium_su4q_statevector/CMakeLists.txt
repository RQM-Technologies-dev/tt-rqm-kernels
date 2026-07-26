cmake_minimum_required(VERSION 3.22)
project(tt_rqm_su4q_statevector LANGUAGES CXX)
find_package(TT-Metalium REQUIRED)
add_executable(tt_rqm_metalium_su4q_statevector_conformance src/statevector_candidate.cpp)
target_compile_features(tt_rqm_metalium_su4q_statevector_conformance PRIVATE cxx_std_20)
target_compile_options(tt_rqm_metalium_su4q_statevector_conformance PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(
    tt_rqm_metalium_su4q_statevector_conformance PRIVATE
    TT_RQM_SV_READER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/gather_reader.cpp"
    TT_RQM_SV_COMPUTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/fused_compute.cpp"
    TT_RQM_SV_WRITER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/scatter_writer.cpp")
target_link_libraries(tt_rqm_metalium_su4q_statevector_conformance PRIVATE TT::Metalium)
