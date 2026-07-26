// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

#ifndef TT_RQM_SU4Q_READER_PATH
#error "TT_RQM_SU4Q_READER_PATH is required"
#endif
#ifndef TT_RQM_SU4Q_LOCAL_PATH
#error "TT_RQM_SU4Q_LOCAL_PATH is required"
#endif
#ifndef TT_RQM_SU4Q_CARTAN_PATH
#error "TT_RQM_SU4Q_CARTAN_PATH is required"
#endif
#ifndef TT_RQM_SU4Q_PHASE_PATH
#error "TT_RQM_SU4Q_PHASE_PATH is required"
#endif
#ifndef TT_RQM_SU4Q_WRITER_PATH
#error "TT_RQM_SU4Q_WRITER_PATH is required"
#endif

namespace {

constexpr std::string_view kProtocol = "tt-rqm-su4q-conformance.v1";
constexpr std::string_view kMetrics = "tt-rqm-su4q-conformance-metrics.v1";
constexpr std::string_view kChainProtocol = "tt-rqm-su4q-chain-conformance.v1";
constexpr std::string_view kChainMetrics = "tt-rqm-su4q-chain-conformance-metrics.v1";
constexpr std::string_view kConvention =
    "rqm-su4q-v1-exp+iabc-qiskit-Klr:kron(q1=Kl,q0=Kr);circuit-little-endian";
constexpr uint32_t kTileElements = 32 * 32;
constexpr uint32_t kTileBytes = kTileElements * sizeof(uint32_t);
constexpr uint32_t kStateLanes = 8;
constexpr uint32_t kBlockLanes = 20;

enum class StageKind { Local, Cartan, Phase };

struct Assignment {
    CoreCoord core;
    uint32_t tiles;
    uint32_t start;
};

struct Stage {
    distributed::MeshWorkload workload;
    uint32_t core_count;
};

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to read " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

void write_text(const std::filesystem::path& path, const std::string& value) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to write " + path.string());
    output << value;
}

std::vector<uint32_t> read_words(const std::filesystem::path& path, size_t count) {
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) != count * sizeof(uint32_t)) {
        throw std::runtime_error(path.filename().string() + " has an unexpected byte count");
    }
    std::vector<uint32_t> words(count);
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(words.data()),
        static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!input) throw std::runtime_error("short read from " + path.string());
    return words;
}

void write_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream output(path, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(words.data()),
        static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!output) throw std::runtime_error("failed to write " + path.string());
}

std::string env_required(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error(std::string("missing environment metadata: ") + name);
    }
    return value;
}

bool env_bool(const char* name) {
    const std::string value = env_required(name);
    if (value == "true") return true;
    if (value == "false") return false;
    throw std::runtime_error(std::string(name) + " must be true or false");
}

std::vector<uint32_t> aos_to_planar(
    const std::vector<uint32_t>& aos,
    uint32_t items,
    uint32_t lanes,
    uint32_t component_tiles) {
    const uint32_t padded = component_tiles * kTileElements;
    std::vector<uint32_t> planar(static_cast<size_t>(lanes) * padded, 0);
    for (uint32_t item = 0; item < items; ++item) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            planar[static_cast<size_t>(lane) * padded + item] =
                aos[static_cast<size_t>(item) * lanes + lane];
        }
    }
    return planar;
}

std::vector<uint32_t> depth_aos_to_planar(
    const std::vector<uint32_t>& aos,
    uint32_t depth,
    uint32_t items,
    uint32_t lanes,
    uint32_t component_tiles) {
    const uint32_t padded = component_tiles * kTileElements;
    std::vector<uint32_t> planar(static_cast<size_t>(depth) * lanes * padded, 0);
    for (uint32_t layer = 0; layer < depth; ++layer) {
        for (uint32_t item = 0; item < items; ++item) {
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                planar[(static_cast<size_t>(layer) * lanes + lane) * padded + item] =
                    aos[(static_cast<size_t>(layer) * items + item) * lanes + lane];
            }
        }
    }
    return planar;
}

std::vector<uint32_t> planar_to_aos(
    const std::vector<uint32_t>& planar,
    uint32_t items,
    uint32_t lanes,
    uint32_t component_tiles) {
    const uint32_t padded = component_tiles * kTileElements;
    if (planar.size() != static_cast<size_t>(lanes) * padded) {
        throw std::runtime_error("planar output size mismatch");
    }
    std::vector<uint32_t> aos(static_cast<size_t>(items) * lanes);
    for (uint32_t item = 0; item < items; ++item) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            aos[static_cast<size_t>(item) * lanes + lane] =
                planar[static_cast<size_t>(lane) * padded + item];
        }
    }
    return aos;
}

void create_cb(Program& program, const CoreRangeSet& cores, uint32_t index) {
    const auto cb = static_cast<tt::CBIndex>(index);
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(kTileBytes, {{cb, DataFormat::Float32}})
            .set_page_size(cb, kTileBytes));
}

ComputeConfig compute_config(const std::vector<uint32_t>& unpack_cbs) {
    std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : unpack_cbs) modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    ComputeConfig config{};
    config.math_fidelity = MathFidelity::HiFi4;
    config.fp32_dest_acc_en = true;
    config.unpack_to_dest_mode = std::move(modes);
    config.math_approx_mode = false;
    return config;
}

std::vector<Assignment> assignments_for(
    const CoreRangeSet& first,
    const CoreRangeSet& second,
    uint32_t first_tiles,
    uint32_t second_tiles,
    uint32_t component_tiles) {
    std::vector<Assignment> assignments;
    uint32_t start = 0;
    for (const auto& [group, per_core] :
         {std::pair{first, first_tiles}, std::pair{second, second_tiles}}) {
        for (const auto& range : group.ranges()) {
            for (const CoreCoord& core : range) {
                assignments.push_back({core, per_core, start});
                start += per_core;
            }
        }
    }
    if (start != component_tiles) throw std::runtime_error("stage work split is incomplete");
    return assignments;
}

Stage build_stage(
    const std::shared_ptr<distributed::MeshDevice>& device,
    const std::shared_ptr<distributed::MeshBuffer>& blocks,
    const std::shared_ptr<distributed::MeshBuffer>& source,
    const std::shared_ptr<distributed::MeshBuffer>& destination,
    uint32_t component_tiles,
    StageKind kind,
    uint32_t block_lane_start,
    uint32_t selector,
    uint32_t block_base_page = 0) {
    const uint32_t parameter_lanes = kind == StageKind::Local ? 4 : 1;
    Program program = CreateProgram();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const auto [core_count, all, group_1, group_2, tiles_1, tiles_2] =
        split_work_to_cores(grid, component_tiles, true);
    for (uint32_t cb = 0; cb < parameter_lanes + kStateLanes; ++cb) create_cb(program, all, cb);
    if (kind != StageKind::Local) {
        create_cb(program, all, 9);
        create_cb(program, all, 10);
    }
    for (uint32_t cb = 16; cb < 16 + kStateLanes; ++cb) create_cb(program, all, cb);

    std::vector<uint32_t> reader_compile;
    TensorAccessorArgs(*blocks).append_to(reader_compile);
    TensorAccessorArgs(*source).append_to(reader_compile);
    std::vector<uint32_t> writer_compile;
    TensorAccessorArgs(*destination).append_to(writer_compile);
    const auto reader = CreateKernel(
        program,
        TT_RQM_SU4Q_READER_PATH,
        all,
        ReaderDataMovementConfig{reader_compile});
    const auto writer = CreateKernel(
        program,
        TT_RQM_SU4Q_WRITER_PATH,
        all,
        WriterDataMovementConfig{writer_compile});
    const char* compute_path = kind == StageKind::Local
        ? TT_RQM_SU4Q_LOCAL_PATH
        : (kind == StageKind::Cartan ? TT_RQM_SU4Q_CARTAN_PATH : TT_RQM_SU4Q_PHASE_PATH);
    std::vector<uint32_t> unpack;
    for (uint32_t cb = 0; cb < parameter_lanes + kStateLanes; ++cb) unpack.push_back(cb);
    if (kind != StageKind::Local) {
        unpack.push_back(9);
        unpack.push_back(10);
    }
    const auto compute =
        CreateKernel(program, compute_path, all, compute_config(unpack));
    for (const auto& item :
         assignments_for(group_1, group_2, tiles_1, tiles_2, component_tiles)) {
        SetRuntimeArgs(
            program,
            reader,
            item.core,
            {
                static_cast<uint32_t>(blocks->address()),
                static_cast<uint32_t>(source->address()),
                item.tiles,
                item.start,
                component_tiles,
                block_lane_start,
                parameter_lanes,
                block_base_page,
            });
        SetRuntimeArgs(program, compute, item.core, {item.tiles, selector});
        SetRuntimeArgs(
            program,
            writer,
            item.core,
            {
                static_cast<uint32_t>(destination->address()),
                item.tiles,
                item.start,
                component_tiles,
            });
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    return {std::move(workload), core_count};
}

}  // namespace

int main() {
    try {
        const std::filesystem::path work_dir(env_required("TT_RQM_SU4Q_DIR"));
        const std::filesystem::path manifest_path(env_required("TT_RQM_SU4Q_MANIFEST"));
        const json manifest = json::parse(read_text(manifest_path));
        const std::string protocol = manifest.value("schema", "");
        const bool chain = protocol == kChainProtocol;
        if ((!chain && protocol != kProtocol) ||
            manifest.value("experiment", "") !=
                (chain ? "su4q-chain-n300-conformance" : "su4q-n300-conformance") ||
            manifest.value("stage", "") != "conformance" ||
            manifest.value("dtype", "") != "float32" ||
            manifest.value("performance_eligible", true) ||
            manifest.value("block_convention_version", "") != kConvention) {
            throw std::runtime_error("unsupported SU4Q conformance manifest");
        }
        const uint32_t items = manifest.at("items").get<uint32_t>();
        const uint32_t depth = chain ? manifest.at("depth").get<uint32_t>() : 1;
        if ((chain && items != 128) || (!chain && items != 128 && items != 4096)) {
            throw std::runtime_error("SU4Q items must be 128 or 4096");
        }
        if (chain && depth != 1 && depth != 8 && depth != 32 && depth != 128) {
            throw std::runtime_error("SU4Q chain depth must be 1, 8, 32, or 128");
        }
        const auto block_shape = manifest.at("block_shape").get<std::vector<uint32_t>>();
        const auto state_shape = manifest.at("state_shape").get<std::vector<uint32_t>>();
        const std::vector<uint32_t> expected_block_shape =
            chain ? std::vector<uint32_t>{depth, items, kBlockLanes}
                  : std::vector<uint32_t>{items, kBlockLanes};
        if (block_shape != expected_block_shape ||
            state_shape != std::vector<uint32_t>{items, kStateLanes}) {
            throw std::runtime_error("invalid SU4Q input shapes");
        }
        const auto& inputs = manifest.at("inputs");
        const auto& outputs = manifest.at("outputs");
        const auto blocks_aos = read_words(
            work_dir / inputs.at("blocks").at("file").get<std::string>(),
            static_cast<size_t>(depth) * items * kBlockLanes);
        const auto states_aos = read_words(
            work_dir / inputs.at("states").at("file").get<std::string>(),
            static_cast<size_t>(items) * kStateLanes);
        const uint32_t component_tiles = (items + kTileElements - 1) / kTileElements;
        const auto packed_blocks = depth_aos_to_planar(
            blocks_aos, depth, items, kBlockLanes, component_tiles);
        const auto packed_states =
            aos_to_planar(states_aos, items, kStateLanes, component_tiles);
        const uint32_t block_bytes = depth * kBlockLanes * component_tiles * kTileBytes;
        const uint32_t state_bytes = kStateLanes * component_tiles * kTileBytes;

        const auto process_start = Clock::now();
        const auto create_start = Clock::now();
        auto device = distributed::MeshDevice::create_unit_mesh(0);
        const double create_s = elapsed(create_start);
        auto& queue = device->mesh_command_queue();
        distributed::DeviceLocalBufferConfig local{};
        local.page_size = kTileBytes;
        local.buffer_type = BufferType::DRAM;
        const auto allocation_start = Clock::now();
        auto blocks = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = block_bytes}, local, device.get());
        auto state_a = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = state_bytes}, local, device.get());
        auto state_b = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = state_bytes}, local, device.get());
        const double allocation_s = elapsed(allocation_start);

        const auto build_start = Clock::now();
        std::vector<Stage> stages;
        for (uint32_t layer = 0; layer < depth; ++layer) {
            const uint32_t base = layer * kBlockLanes * component_tiles;
            stages.push_back(build_stage(
                device, blocks, state_a, state_b, component_tiles, StageKind::Local, 11, 0, base));
            stages.push_back(build_stage(
                device, blocks, state_b, state_a, component_tiles, StageKind::Local, 15, 1, base));
            stages.push_back(build_stage(
                device, blocks, state_a, state_b, component_tiles, StageKind::Cartan, 8, 0, base));
            stages.push_back(build_stage(
                device, blocks, state_b, state_a, component_tiles, StageKind::Cartan, 9, 1, base));
            stages.push_back(build_stage(
                device, blocks, state_a, state_b, component_tiles, StageKind::Cartan, 10, 2, base));
            stages.push_back(build_stage(
                device, blocks, state_b, state_a, component_tiles, StageKind::Local, 0, 0, base));
            stages.push_back(build_stage(
                device, blocks, state_a, state_b, component_tiles, StageKind::Local, 4, 1, base));
            stages.push_back(build_stage(
                device, blocks, state_b, state_a, component_tiles, StageKind::Phase, 19, 0, base));
        }
        const double build_s = elapsed(build_start);

        const auto h2d_start = Clock::now();
        distributed::EnqueueWriteMeshBuffer(queue, blocks, packed_blocks, false);
        distributed::EnqueueWriteMeshBuffer(queue, state_a, packed_states, false);
        distributed::Finish(queue);
        const double h2d_s = elapsed(h2d_start);
        const auto execute_start = Clock::now();
        for (auto& stage : stages) {
            distributed::EnqueueMeshWorkload(queue, stage.workload, false);
        }
        distributed::Finish(queue);
        const double execute_s = elapsed(execute_start);
        const auto d2h_start = Clock::now();
        std::vector<uint32_t> packed_output;
        distributed::EnqueueReadMeshBuffer(queue, packed_output, state_a, true);
        distributed::Finish(queue);
        const double d2h_s = elapsed(d2h_start);
        const auto output_aos =
            planar_to_aos(packed_output, items, kStateLanes, component_tiles);
        write_words(
            work_dir / outputs.at("states").get<std::string>(),
            output_aos);
        const auto close_start = Clock::now();
        if (!device->close()) throw std::runtime_error("failed to close MeshDevice");
        const double close_s = elapsed(close_start);

        std::vector<uint32_t> core_counts;
        for (const auto& stage : stages) core_counts.push_back(stage.core_count);
        const json metadata = {
            {"implementation_class", chain
                ? "depth_chained_device_resident_su4q"
                : "eight_program_device_resident_su4q"},
            {"candidate_sha256", env_required("TT_RQM_SU4Q_CANDIDATE_SHA256")},
            {"source_commit", env_required("TT_RQM_SU4Q_SOURCE_COMMIT")},
            {"source_tree_clean", env_bool("TT_RQM_SU4Q_SOURCE_TREE_CLEAN")},
            {"source_bundle_sha256", env_required("TT_RQM_SU4Q_SOURCE_BUNDLE_SHA256")},
            {"tt_rqm_base_commit", env_required("TT_RQM_SU4Q_BASE_COMMIT")},
            {"tt_metal_commit", env_required("TT_RQM_SU4Q_TT_METAL_COMMIT")},
            {"compiler_version", env_required("TT_RQM_SU4Q_COMPILER_VERSION")},
            {"runtime_version", env_required("TT_RQM_SU4Q_RUNTIME_VERSION")},
            {"block_convention_version", kConvention},
            {"device_arch", "wormhole_b0"},
            {"device_count", 1},
            {"device_id", 0},
            {"device_create_count", 1},
            {"device_close_count", 1},
            {"program_count", 8 * depth},
            {"depth", depth},
            {"stage_core_counts", core_counts},
            {"stage_order", {
                "right_q0", "right_q1", "cartan_xx", "cartan_yy",
                "cartan_zz", "left_q0", "left_q1", "global_phase"}},
            {"local_factor_order", "SU(2)(q1) tensor SU(2)(q0)"},
            {"cartan_convention", "exp[i(aXX+bYY+cZZ)]"},
            {"basis_order", {"00", "01", "10", "11"}},
            {"intermediate_storage", "device_dram_ping_pong"},
            {"intermediate_d2h_count", 0},
            {"intermediate_h2d_count", 0},
            {"initial_block_upload_count", 1},
            {"initial_state_upload_count", 1},
            {"final_state_download_count", 1},
            {"automatic_normalization", false},
        };
        const json metrics = {
            {"schema", chain ? kChainMetrics : kMetrics},
            {"protocol", protocol},
            {"experiment", chain ? "su4q-chain-n300-conformance" : "su4q-n300-conformance"},
            {"stage", "conformance"},
            {"dtype", "float32"},
            {"execution_label", "hardware"},
            {"items", items},
            {"depth", depth},
            {"block_shape", block_shape},
            {"state_shape", state_shape},
            {"output_shape", std::vector<uint32_t>{items, kStateLanes}},
            {"performance_eligible", false},
            {"stable_benchmark", false},
            {"claim_level", nullptr},
            {"timings_s", {
                {"device_create", create_s},
                {"buffer_allocation", allocation_s},
                {"program_build", build_s},
                {"h2d", h2d_s},
                {"device_execute_diagnostic_only", execute_s},
                {"d2h", d2h_s},
                {"device_close", close_s},
                {"candidate_process", elapsed(process_start)},
            }},
            {"candidate_metadata", metadata},
        };
        write_text(
            work_dir / outputs.at("metrics").get<std::string>(),
            metrics.dump(2) + "\n");
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "tt_rqm_metalium_su4q_conformance failed: " << exc.what() << "\n";
        return 2;
    }
}
