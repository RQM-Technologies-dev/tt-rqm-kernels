if(NOT DEFINED TT_RQM_SV_SOURCE_DIR)
    message(FATAL_ERROR "TT_RQM_SV_SOURCE_DIR is required")
endif()
function(tt_rqm_register_su4q_statevector)
    add_executable(tt_rqm_metalium_su4q_statevector_conformance)
    target_sources(tt_rqm_metalium_su4q_statevector_conformance PRIVATE
        "${TT_RQM_SV_SOURCE_DIR}/src/statevector_candidate.cpp")
    target_compile_features(tt_rqm_metalium_su4q_statevector_conformance PRIVATE cxx_std_20)
    target_compile_options(tt_rqm_metalium_su4q_statevector_conformance PRIVATE -Wall -Wextra -Werror)
    target_compile_definitions(tt_rqm_metalium_su4q_statevector_conformance PRIVATE
        TT_RQM_SV_READER_PATH="${TT_RQM_SV_SOURCE_DIR}/kernels/gather_reader.cpp"
        TT_RQM_SV_COMPUTE_PATH="${TT_RQM_SV_SOURCE_DIR}/kernels/fused_compute.cpp"
        TT_RQM_SV_WRITER_PATH="${TT_RQM_SV_SOURCE_DIR}/kernels/scatter_writer.cpp")
    target_link_libraries(tt_rqm_metalium_su4q_statevector_conformance PRIVATE TT::Metalium)
    if(TARGET TT::CommonPCH)
        tt_reuse_precompile_headers(tt_rqm_metalium_su4q_statevector_conformance TT::CommonPCH)
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL tt_rqm_register_su4q_statevector)
