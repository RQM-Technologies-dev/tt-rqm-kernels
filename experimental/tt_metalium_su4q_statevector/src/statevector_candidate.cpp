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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

#ifndef TT_RQM_SV_READER_PATH
#error "TT_RQM_SV_READER_PATH is required"
#endif
#ifndef TT_RQM_SV_COMPUTE_PATH
#error "TT_RQM_SV_COMPUTE_PATH is required"
#endif
#ifndef TT_RQM_SV_WRITER_PATH
#error "TT_RQM_SV_WRITER_PATH is required"
#endif

namespace {
constexpr std::string_view kProtocol = "tt-rqm-su4q-statevector-conformance.v1";
constexpr std::string_view kMetrics = "tt-rqm-su4q-statevector-conformance-metrics.v1";
constexpr std::string_view kConvention =
    "rqm-su4q-v1-exp+iabc-qiskit-Klr:kron(q1=Kl,q0=Kr);circuit-little-endian";
constexpr uint32_t kTileElements = 1024;
constexpr uint32_t kTileBytes = kTileElements * sizeof(uint32_t);
constexpr uint32_t kBatch = 4;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string env_required(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') throw std::runtime_error(std::string("missing ") + name);
    return value;
}
bool env_bool(const char* name) {
    const std::string value = env_required(name);
    if (value == "true") return true;
    if (value == "false") return false;
    throw std::runtime_error(std::string(name) + " must be boolean");
}
std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to read " + path.string());
    std::ostringstream out; out << input.rdbuf(); return out.str();
}
void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to write " + path.string());
    output << text;
}
std::vector<uint32_t> read_words(const std::filesystem::path& path, size_t words) {
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) != words * sizeof(uint32_t)) {
        throw std::runtime_error(path.filename().string() + " byte count mismatch");
    }
    std::vector<uint32_t> value(words);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(value.data()), static_cast<std::streamsize>(words * 4));
    if (!input) throw std::runtime_error("short read");
    return value;
}
void write_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    if (!output) throw std::runtime_error("failed output write");
}
void create_cb(
    Program& program, const CoreRangeSet& cores, uint32_t index, uint32_t pages = 1) {
    const auto cb = static_cast<tt::CBIndex>(index);
    CreateCircularBuffer(
        program, cores,
        CircularBufferConfig(pages * kTileBytes, {{cb, DataFormat::Float32}})
            .set_page_size(cb, kTileBytes));
}
std::vector<uint32_t> pack_states(
    const std::vector<uint32_t>& aos, uint32_t total_amplitudes, uint32_t component_tiles) {
    std::vector<uint32_t> planar(static_cast<size_t>(2) * component_tiles * kTileElements, 0);
    for (uint32_t index = 0; index < total_amplitudes; ++index) {
        planar[index] = aos[2 * index];
        planar[component_tiles * kTileElements + index] = aos[2 * index + 1];
    }
    return planar;
}
std::vector<uint32_t> unpack_states(
    const std::vector<uint32_t>& planar, uint32_t total_amplitudes, uint32_t component_tiles) {
    std::vector<uint32_t> aos(static_cast<size_t>(total_amplitudes) * 2);
    for (uint32_t index = 0; index < total_amplitudes; ++index) {
        aos[2 * index] = planar[index];
        aos[2 * index + 1] = planar[component_tiles * kTileElements + index];
    }
    return aos;
}
std::vector<uint32_t> broadcast_blocks(
    const std::vector<uint32_t>& blocks, uint32_t depth, uint32_t work_tiles) {
    std::vector<uint32_t> packed(
        static_cast<size_t>(depth) * 20 * work_tiles * kTileElements);
    for (uint32_t layer = 0; layer < depth; ++layer) {
        for (uint32_t lane = 0; lane < 20; ++lane) {
            const uint32_t value = blocks[layer * 20 + lane];
            const size_t start =
                (static_cast<size_t>(layer) * 20 + lane) * work_tiles * kTileElements;
            std::fill_n(packed.begin() + start, work_tiles * kTileElements, value);
        }
    }
    return packed;
}
struct Assignment { CoreCoord core; uint32_t tiles; uint32_t start; };
std::vector<Assignment> assignments_for(
    const CoreRangeSet& first, const CoreRangeSet& second,
    uint32_t first_tiles, uint32_t second_tiles, uint32_t total) {
    std::vector<Assignment> result;
    uint32_t start = 0;
    for (const auto& [group, count] :
         {std::pair{first, first_tiles}, std::pair{second, second_tiles}}) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                result.push_back({core, count, start}); start += count;
            }
        }
    }
    if (start != total) throw std::runtime_error("incomplete work split");
    return result;
}
struct Stage { distributed::MeshWorkload workload; uint32_t cores; };
Stage build_stage(
    const std::shared_ptr<distributed::MeshDevice>& device,
    const std::shared_ptr<distributed::MeshBuffer>& blocks,
    const std::shared_ptr<distributed::MeshBuffer>& source,
    const std::shared_ptr<distributed::MeshBuffer>& destination,
    uint32_t work_tiles, uint32_t total_amplitudes, uint32_t amplitudes_per_state,
    uint32_t state_tiles, uint32_t layer, uint32_t q0, uint32_t q1) {
    Program program = CreateProgram();
    const auto [core_count, all, g1, g2, t1, t2] =
        split_work_to_cores(device->compute_with_storage_grid_size(), work_tiles, true);
    for (uint32_t cb = 0; cb <= 13; ++cb) create_cb(program, all, cb);
    create_cb(program, all, 14, 64);
    create_cb(program, all, 15, 2);
    for (uint32_t cb = 16; cb <= 31; ++cb) create_cb(program, all, cb);
    std::vector<uint32_t> reader_compile;
    TensorAccessorArgs(*source).append_to(reader_compile);
    TensorAccessorArgs(*blocks).append_to(reader_compile);
    std::vector<uint32_t> writer_compile;
    TensorAccessorArgs(*destination).append_to(writer_compile);
    const auto reader = CreateKernel(program, TT_RQM_SV_READER_PATH, all, ReaderDataMovementConfig{reader_compile});
    const auto writer = CreateKernel(program, TT_RQM_SV_WRITER_PATH, all, WriterDataMovementConfig{writer_compile});
    std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb = 0; cb <= 23; ++cb) modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    ComputeConfig config{};
    config.math_fidelity = MathFidelity::HiFi4;
    config.fp32_dest_acc_en = true;
    config.unpack_to_dest_mode = std::move(modes);
    config.math_approx_mode = false;
    const auto compute = CreateKernel(program, TT_RQM_SV_COMPUTE_PATH, all, config);
    for (const auto& item : assignments_for(g1, g2, t1, t2, work_tiles)) {
        SetRuntimeArgs(program, reader, item.core, {
            static_cast<uint32_t>(source->address()), static_cast<uint32_t>(blocks->address()),
            item.tiles, item.start, total_amplitudes, amplitudes_per_state,
            state_tiles, work_tiles, layer, q0, q1});
        SetRuntimeArgs(program, compute, item.core, {item.tiles});
        SetRuntimeArgs(program, writer, item.core, {
            static_cast<uint32_t>(destination->address()), item.tiles, item.start,
            total_amplitudes, amplitudes_per_state, state_tiles, q0, q1});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    return {std::move(workload), core_count};
}
}  // namespace

int main() {
    try {
        const std::filesystem::path work_dir(env_required("TT_RQM_SV_DIR"));
        const json manifest = json::parse(read_text(env_required("TT_RQM_SV_MANIFEST")));
        if (manifest.value("schema", "") != kProtocol ||
            manifest.value("experiment", "") != "su4q-multiqubit-n300-conformance" ||
            manifest.value("performance_eligible", true) ||
            manifest.value("block_convention_version", "") != kConvention) {
            throw std::runtime_error("unsupported statevector manifest");
        }
        const uint32_t qubits = manifest.at("qubits");
        const uint32_t depth = manifest.at("depth");
        if (qubits != 3 && qubits != 4 && qubits != 6 && qubits != 10 && qubits != 15)
            throw std::runtime_error("unsupported qubit count");
        if (depth != 1 && depth != 8 && depth != 32 && depth != 128)
            throw std::runtime_error("unsupported depth");
        const uint32_t amplitudes = 1U << qubits;
        const uint32_t total_amplitudes = kBatch * amplitudes;
        const uint32_t state_tiles = (total_amplitudes + kTileElements - 1) / kTileElements;
        if (manifest.at("block_shape") != json::array({depth, 20}) ||
            manifest.at("pair_shape") != json::array({depth, 2}) ||
            manifest.at("state_shape") != json::array({kBatch, amplitudes, 2})) {
            throw std::runtime_error("statevector shape mismatch");
        }
        const auto& inputs = manifest.at("inputs");
        const auto& outputs = manifest.at("outputs");
        const auto block_words = read_words(
            work_dir / inputs.at("blocks").at("file").get<std::string>(), depth * 20);
        const auto pair_words = read_words(
            work_dir / inputs.at("pairs").at("file").get<std::string>(), depth * 2);
        const auto state_words = read_words(
            work_dir / inputs.at("states").at("file").get<std::string>(),
            static_cast<size_t>(total_amplitudes) * 2);
        for (uint32_t layer = 0; layer < depth; ++layer) {
            const uint32_t q0 = pair_words[2 * layer], q1 = pair_words[2 * layer + 1];
            if (q0 >= qubits || q1 >= qubits || q0 == q1) throw std::runtime_error("invalid target pair");
        }
        const auto packed_blocks = broadcast_blocks(block_words, depth, state_tiles);
        const auto packed_states = pack_states(state_words, total_amplitudes, state_tiles);
        const auto process_start = Clock::now();
        const auto create_start = Clock::now();
        auto device = distributed::MeshDevice::create_unit_mesh(0);
        const double create_s = elapsed(create_start);
        auto& queue = device->mesh_command_queue();
        distributed::DeviceLocalBufferConfig local{};
        local.page_size = kTileBytes;
        local.buffer_type = BufferType::DRAM;
        auto blocks = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = packed_blocks.size() * 4}, local, device.get());
        auto state_a = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = packed_states.size() * 4}, local, device.get());
        auto state_b = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = packed_states.size() * 4}, local, device.get());
        const auto build_start = Clock::now();
        std::vector<Stage> stages;
        for (uint32_t layer = 0; layer < depth; ++layer) {
            auto source = layer % 2 == 0 ? state_a : state_b;
            auto destination = layer % 2 == 0 ? state_b : state_a;
            stages.push_back(build_stage(
                device, blocks, source, destination, state_tiles, total_amplitudes,
                amplitudes, state_tiles, layer,
                pair_words[2 * layer], pair_words[2 * layer + 1]));
        }
        const double build_s = elapsed(build_start);
        const auto h2d_start = Clock::now();
        distributed::EnqueueWriteMeshBuffer(queue, blocks, packed_blocks, false);
        distributed::EnqueueWriteMeshBuffer(queue, state_a, packed_states, false);
        distributed::Finish(queue);
        const double h2d_s = elapsed(h2d_start);
        const auto execute_start = Clock::now();
        for (auto& stage : stages) distributed::EnqueueMeshWorkload(queue, stage.workload, false);
        distributed::Finish(queue);
        const double execute_s = elapsed(execute_start);
        const auto d2h_start = Clock::now();
        std::vector<uint32_t> packed_output;
        distributed::EnqueueReadMeshBuffer(queue, packed_output, depth % 2 == 0 ? state_a : state_b, true);
        distributed::Finish(queue);
        const double d2h_s = elapsed(d2h_start);
        write_words(
            work_dir / outputs.at("states").get<std::string>(),
            unpack_states(packed_output, total_amplitudes, state_tiles));
        const auto close_start = Clock::now();
        if (!device->close()) throw std::runtime_error("device close failed");
        const double close_s = elapsed(close_start);
        std::vector<uint32_t> core_counts;
        for (const auto& stage : stages) core_counts.push_back(stage.cores);
        const json metadata = {
            {"implementation_class", "fused_su4q_arbitrary_pair_device_resident_statevector"},
            {"candidate_sha256", env_required("TT_RQM_SV_CANDIDATE_SHA256")},
            {"source_bundle_sha256", env_required("TT_RQM_SV_SOURCE_BUNDLE_SHA256")},
            {"source_commit", env_required("TT_RQM_SV_SOURCE_COMMIT")},
            {"source_tree_clean", env_bool("TT_RQM_SV_SOURCE_TREE_CLEAN")},
            {"tt_metal_commit", env_required("TT_RQM_SV_TT_METAL_COMMIT")},
            {"device_arch", "wormhole_b0"}, {"device_id", 0}, {"device_count", 1},
            {"device_create_count", 1}, {"device_close_count", 1},
            {"program_count", depth}, {"dispatch_count", depth},
            {"stage_core_counts", core_counts},
            {"initial_block_upload_count", 1}, {"initial_state_upload_count", 1},
            {"intermediate_h2d_count", 0}, {"intermediate_d2h_count", 0},
            {"final_state_download_count", 1},
            {"quartet_order", "|q1 q0>"}, {"automatic_normalization", false},
        };
        const json metrics = {
            {"schema", kMetrics}, {"protocol", kProtocol},
            {"experiment", "su4q-multiqubit-n300-conformance"},
            {"performance_eligible", false}, {"stable_benchmark", false}, {"claim_level", nullptr},
            {"qubits", qubits}, {"depth", depth}, {"batch", kBatch},
            {"timings_s", {{"device_create", create_s}, {"program_build", build_s},
                {"h2d", h2d_s}, {"device_execute_diagnostic_only", execute_s},
                {"d2h", d2h_s}, {"device_close", close_s},
                {"candidate_process", elapsed(process_start)}}},
            {"candidate_metadata", metadata},
        };
        write_text(work_dir / outputs.at("metrics").get<std::string>(), metrics.dump(2) + "\n");
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "su4q statevector candidate failed: " << exc.what() << "\n";
        return 2;
    }
}
