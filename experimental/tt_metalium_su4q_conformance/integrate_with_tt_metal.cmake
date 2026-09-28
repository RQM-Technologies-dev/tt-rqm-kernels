# Register the private SU4Q target after tt-metal has defined TT::Metalium.
if(NOT DEFINED TT_RQM_SU4Q_SOURCE_DIR)
    message(FATAL_ERROR "TT_RQM_SU4Q_SOURCE_DIR is required")
endif()

function(tt_rqm_register_su4q_target)
    add_executable(tt_rqm_metalium_su4q_conformance)
    target_sources(
        tt_rqm_metalium_su4q_conformance
        PRIVATE "${TT_RQM_SU4Q_SOURCE_DIR}/src/su4q_conformance_candidate.cpp"
    )
    target_compile_features(tt_rqm_metalium_su4q_conformance PRIVATE cxx_std_20)
    target_compile_options(tt_rqm_metalium_su4q_conformance PRIVATE -Wall -Wextra -Werror)
    target_compile_definitions(
        tt_rqm_metalium_su4q_conformance
        PRIVATE
            TT_RQM_SU4Q_READER_PATH="${TT_RQM_SU4Q_SOURCE_DIR}/kernels/reader_stage.cpp"
            TT_RQM_SU4Q_LOCAL_PATH="${TT_RQM_SU4Q_SOURCE_DIR}/kernels/compute_local.cpp"
            TT_RQM_SU4Q_CARTAN_PATH="${TT_RQM_SU4Q_SOURCE_DIR}/kernels/compute_cartan.cpp"
            TT_RQM_SU4Q_PHASE_PATH="${TT_RQM_SU4Q_SOURCE_DIR}/kernels/compute_phase.cpp"
            TT_RQM_SU4Q_WRITER_PATH="${TT_RQM_SU4Q_SOURCE_DIR}/kernels/writer_state.cpp"
    )
    target_link_libraries(tt_rqm_metalium_su4q_conformance PRIVATE TT::Metalium)
    if(TARGET TT::CommonPCH)
        tt_reuse_precompile_headers(tt_rqm_metalium_su4q_conformance TT::CommonPCH)
    endif()
endfunction()

cmake_language(
    DEFER
    DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL tt_rqm_register_su4q_target
)
