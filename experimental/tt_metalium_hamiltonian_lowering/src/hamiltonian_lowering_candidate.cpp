// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::string_view kProtocol = "tt-rqm-external-hamiltonian-lowering.v1";
constexpr std::string_view kMetrics = "tt-rqm-external-hamiltonian-lowering-metrics.v1";
constexpr uint32_t kTileElements = 32 * 32;
constexpr uint32_t kTileBytes = kTileElements * sizeof(uint32_t);
constexpr uint32_t kPlanes = 6;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to read " + path.string());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
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
    input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!input) throw std::runtime_error("short read from " + path.string());
    return words;
}

void write_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!output) throw std::runtime_error("failed to write " + path.string());
}

std::string env_required(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') throw std::runtime_error(std::string("missing environment metadata: ") + name);
    return value;
}

bool env_bool(const char* name) {
    const std::string value = env_required(name);
    if (value == "true") return true;
    if (value == "false") return false;
    throw std::runtime_error(std::string(name) + " must be true or false");
}

void create_cb(Program& program, const CoreRangeSet& core, uint32_t index) {
    const auto cb = static_cast<tt::CBIndex>(index);
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(kTileBytes, {{cb, DataFormat::Float32}}).set_page_size(cb, kTileBytes));
}

std::vector<uint32_t> pack_inputs(
    const std::vector<uint32_t>& coefficients,
    const std::vector<uint32_t>& dt,
    uint32_t count,
    uint32_t component_tiles,
    float inverse_hbar) {
    const uint32_t padded = component_tiles * kTileElements;
    std::vector<uint32_t> packed(static_cast<size_t>(kPlanes) * padded, 0);
    for (uint32_t row = 0; row < count; ++row) {
        for (uint32_t lane = 0; lane < 4; ++lane) {
            packed[static_cast<size_t>(lane) * padded + row] = coefficients[static_cast<size_t>(row) * 4 + lane];
        }
        packed[static_cast<size_t>(4) * padded + row] = dt.size() == 1 ? dt[0] : dt[row];
        packed[static_cast<size_t>(5) * padded + row] = std::bit_cast<uint32_t>(inverse_hbar);
    }
    return packed;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> unpack_outputs(
    const std::vector<uint32_t>& packed, uint32_t count, uint32_t component_tiles) {
    const uint32_t padded = component_tiles * kTileElements;
    if (packed.size() != static_cast<size_t>(kPlanes) * padded) throw std::runtime_error("output size mismatch");
    std::vector<uint32_t> rotors(static_cast<size_t>(count) * 4);
    std::vector<uint32_t> phases(static_cast<size_t>(count) * 2);
    for (uint32_t row = 0; row < count; ++row) {
        for (uint32_t lane = 0; lane < 4; ++lane) rotors[static_cast<size_t>(row) * 4 + lane] = packed[static_cast<size_t>(lane) * padded + row];
        for (uint32_t lane = 0; lane < 2; ++lane) phases[static_cast<size_t>(row) * 2 + lane] = packed[static_cast<size_t>(4 + lane) * padded + row];
    }
    return {std::move(rotors), std::move(phases)};
}

}  // namespace

int main() {
    try {
        const char* work_value = std::getenv("TT_RQM_H2A_DIR");
        const char* manifest_value = std::getenv("TT_RQM_H2A_MANIFEST");
        if (work_value == nullptr || manifest_value == nullptr) throw std::runtime_error("TT_RQM_H2A_DIR and TT_RQM_H2A_MANIFEST are required");
        const std::filesystem::path work_dir(work_value);
        const json manifest = json::parse(read_text(manifest_value));
        if (manifest.value("schema", "") != kProtocol || manifest.value("benchmark", "") != "HamiltonianLoweringBench" ||
            manifest.value("stage", "") != "conformance" || manifest.value("dtype", "") != "float32") {
            throw std::runtime_error("unsupported H2A external manifest");
        }
        const auto shape = manifest.at("hamiltonian_shape").get<std::vector<uint32_t>>();
        if (shape.size() != 3 || shape[2] != 4 || shape[0] == 0 || shape[1] == 0) throw std::runtime_error("invalid Hamiltonian shape");
        const uint32_t count = shape[0] * shape[1];
        const auto dt_shape = manifest.at("dt_shape").get<std::vector<uint32_t>>();
        if (!(dt_shape.empty() || dt_shape == std::vector<uint32_t>{shape[0], shape[1]})) throw std::runtime_error("dt must be scalar or exactly [B,K]");
        const auto& inputs = manifest.at("inputs");
        const auto& outputs = manifest.at("outputs");
        const bool primitive_probe = std::getenv("TT_RQM_H2A_PRIMITIVE_PROBE_ONLY") != nullptr;
        const auto coefficients = read_words(work_dir / inputs.at("hamiltonians").get<std::string>(), static_cast<size_t>(count) * 4);
        const auto dt = read_words(work_dir / inputs.at("dt").get<std::string>(), dt_shape.empty() ? 1 : count);
        const float hbar = manifest.at("hbar").get<float>();
        if (!(hbar > 0.0f)) throw std::runtime_error("hbar must be positive");

        const uint32_t component_tiles = (count + kTileElements - 1) / kTileElements;
        const auto packed = pack_inputs(coefficients, dt, count, component_tiles, 1.0f / hbar);
        const uint32_t buffer_bytes = kPlanes * component_tiles * kTileBytes;
        const auto process_start = Clock::now();
        const auto create_start = Clock::now();
        auto device = distributed::MeshDevice::create_unit_mesh(0);
        const double create_s = elapsed(create_start);
        auto& queue = device->mesh_command_queue();
        distributed::DeviceLocalBufferConfig local{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
        auto input = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = buffer_bytes}, local, device.get());
        auto output = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = buffer_bytes}, local, device.get());

        Program program = CreateProgram();
        const CoreCoord core_coord{0, 0};
        const CoreRangeSet core{CoreRange(core_coord, core_coord)};
        for (uint32_t cb = 0; cb < 32; ++cb) create_cb(program, core, cb);
        std::vector<uint32_t> reader_compile;
        TensorAccessorArgs(*input).append_to(reader_compile);
        std::vector<uint32_t> writer_compile;
        TensorAccessorArgs(*output).append_to(writer_compile);
        const auto reader = CreateKernel(program, TT_RQM_H2A_READER_PATH, core, DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile});
        const auto writer = CreateKernel(program, TT_RQM_H2A_WRITER_PATH, core, DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile});
        std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::UnpackToDestFp32);
        const auto compute = CreateKernel(program, primitive_probe ? TT_RQM_H2A_PRIMITIVE_PROBE_PATH : TT_RQM_H2A_COMPUTE_PATH, core, ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = modes,
            .math_approx_mode = false,
        });
        SetRuntimeArgs(program, reader, core_coord, {input->address(), component_tiles, component_tiles});
        SetRuntimeArgs(program, compute, core_coord, {component_tiles});
        SetRuntimeArgs(program, writer, core_coord, {output->address(), component_tiles, component_tiles});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));

        const auto h2d_start = Clock::now();
        distributed::EnqueueWriteMeshBuffer(queue, input, packed, false);
        distributed::Finish(queue);
        const double h2d_s = elapsed(h2d_start);
        const auto execute_start = Clock::now();
        distributed::EnqueueMeshWorkload(queue, workload, false);
        distributed::Finish(queue);
        const double execute_s = elapsed(execute_start);
        const auto d2h_start = Clock::now();
        std::vector<uint32_t> packed_output;
        distributed::EnqueueReadMeshBuffer(queue, packed_output, output, true);
        distributed::Finish(queue);
        const double d2h_s = elapsed(d2h_start);
        const auto [rotors, phases] = unpack_outputs(packed_output, count, component_tiles);
        write_words(work_dir / outputs.at("rotors").get<std::string>(), rotors);
        write_words(work_dir / outputs.at("phases").get<std::string>(), phases);
        const auto close_start = Clock::now();
        if (!device->close()) throw std::runtime_error("failed to close MeshDevice");
        const double close_s = elapsed(close_start);

        const json metadata = {
            {"implementation_class", "single_core_tensix_sfpu_h2a"},
            {"candidate_sha256", env_required("TT_RQM_H2A_CANDIDATE_SHA256")},
            {"source_commit", env_required("TT_RQM_H2A_SOURCE_COMMIT")},
            {"source_tree_clean", env_bool("TT_RQM_H2A_SOURCE_TREE_CLEAN")},
            {"source_bundle_sha256", env_required("TT_RQM_H2A_SOURCE_BUNDLE_SHA256")},
            {"tt_metal_commit", env_required("TT_RQM_H2A_TT_METAL_COMMIT")},
            {"compiler_version", env_required("TT_RQM_H2A_COMPILER_VERSION")},
            {"runtime_version", env_required("TT_RQM_H2A_RUNTIME_VERSION")},
            {"device_count", 1}, {"device_id", 0}, {"device_arch", "wormhole_b0"}, {"core_count", 1},
            {"arithmetic_path", "single Tensix compute core with FP32 SFPU"},
            {"input_layout", "component-planar Float32 tiles; row-major [B,K] lanes"},
            {"output_layout", "six component-planar Float32 tiles restored to row-major [B,K,*]"},
            {"scalar_dt_expansion", dt_shape.empty()},
            {"sfpu_sqrt_mode", "sqrt_tile<false>; fp32 destination"},
            {"sfpu_reciprocal_mode", "recip_tile<false> on lane-wise nonzero safe_r"},
            {"sfpu_sine_mode", "native Wormhole sin_tile with pinned four-stage Cody-Waite reduction"},
            {"sfpu_cosine_mode", "native Wormhole cos_tile with pinned four-stage Cody-Waite reduction"},
            {"zero_mask_strategy", "eqz(r2), SFPI FP32 select r->1 before reciprocal, identity output select"},
            {"device_create_count", 1}, {"device_close_count", 1}
        };
        const json metrics = {
            {"schema", kMetrics}, {"protocol", kProtocol}, {"benchmark", "HamiltonianLoweringBench"},
            {"stage", "conformance"}, {"dtype", "float32"}, {"execution_label", "hardware"},
            {"hamiltonian_shape", shape}, {"dt_shape", dt_shape},
            {"stable_benchmark", false}, {"performance_eligible", false}, {"claim_level", nullptr},
            {"timings_s", {{"device_create", create_s}, {"h2d", h2d_s}, {"diagnostic_device_execute", execute_s},
                {"d2h", d2h_s}, {"device_close", close_s}, {"candidate_process", elapsed(process_start)}}},
            {"candidate_metadata", metadata}
        };
        write_text(work_dir / outputs.at("metrics").get<std::string>(), metrics.dump(2) + "\n");
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "tt_rqm_metalium_hamiltonian_lowering_candidate failed: " << exc.what() << "\n";
        return 2;
    }
}
