// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

namespace {

constexpr std::string_view kProtocol = "tt-rqm-external-hamiltonian-evolution.v1";
constexpr std::string_view kMetrics = "tt-rqm-external-hamiltonian-evolution-metrics.v1";
constexpr uint32_t kLanes = 6;
constexpr uint32_t kTileElements = 32 * 32;
constexpr uint32_t kTileBytes = kTileElements * sizeof(uint32_t);

struct Assignment {
    CoreCoord core;
    uint32_t tiles;
    uint32_t start;
};

struct PreparedProgram {
    distributed::MeshWorkload workload;
    uint32_t core_count;
};

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

std::vector<uint32_t> pack_h2b_input(
    const std::vector<uint32_t>& coefficients,
    const std::vector<uint32_t>& dt,
    uint32_t batch,
    uint32_t steps,
    uint32_t component_tiles,
    float inverse_hbar) {
    const uint32_t padded = component_tiles * kTileElements;
    std::vector<uint32_t> packed(static_cast<size_t>(steps) * kLanes * padded, 0);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t step = 0; step < steps; ++step) {
            const uint32_t row = b * steps + step;
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const size_t page_offset = (static_cast<size_t>(step) * kLanes + lane) * padded;
                packed[page_offset + b] = coefficients[static_cast<size_t>(row) * 4 + lane];
            }
            packed[(static_cast<size_t>(step) * kLanes + 4) * padded + b] =
                dt.size() == 1 ? dt[0] : dt[row];
            packed[(static_cast<size_t>(step) * kLanes + 5) * padded + b] =
                std::bit_cast<uint32_t>(inverse_hbar);
        }
    }
    return packed;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> unpack_final_output(
    const std::vector<uint32_t>& packed, uint32_t batch, uint32_t component_tiles) {
    const uint32_t padded = component_tiles * kTileElements;
    if (packed.size() != static_cast<size_t>(kLanes) * padded) {
        throw std::runtime_error("final output size mismatch");
    }
    std::vector<uint32_t> rotors(static_cast<size_t>(batch) * 4);
    std::vector<uint32_t> phases(static_cast<size_t>(batch) * 2);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t lane = 0; lane < 4; ++lane) {
            rotors[static_cast<size_t>(b) * 4 + lane] = packed[static_cast<size_t>(lane) * padded + b];
        }
        for (uint32_t lane = 0; lane < 2; ++lane) {
            phases[static_cast<size_t>(b) * 2 + lane] =
                packed[static_cast<size_t>(4 + lane) * padded + b];
        }
    }
    return {std::move(rotors), std::move(phases)};
}

void create_cb(Program& program, const CoreRangeSet& cores, uint32_t index, uint32_t depth = 1) {
    const auto cb = static_cast<tt::CBIndex>(index);
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(depth * kTileBytes, {{cb, DataFormat::Float32}})
            .set_page_size(cb, kTileBytes));
}

ComputeConfig compute_config(const std::vector<uint32_t>& unpack_cbs) {
    std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : unpack_cbs) modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    return ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = modes,
        .math_approx_mode = false,
    };
}

PreparedProgram build_h2a_program(
    const std::shared_ptr<distributed::MeshDevice>& device,
    const std::shared_ptr<distributed::MeshBuffer>& input,
    const std::shared_ptr<distributed::MeshBuffer>& intermediate,
    uint32_t component_tiles,
    uint32_t steps) {
    Program program = CreateProgram();
    const CoreCoord core_coord{0, 0};
    const CoreRangeSet core{CoreRange(core_coord, core_coord)};
    for (uint32_t cb = 0; cb < 32; ++cb) create_cb(program, core, cb);
    std::vector<uint32_t> reader_compile;
    TensorAccessorArgs(*input).append_to(reader_compile);
    std::vector<uint32_t> writer_compile;
    TensorAccessorArgs(*intermediate).append_to(writer_compile);
    const auto reader = CreateKernel(program, TT_RQM_H2B_H2A_READER_PATH, core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile});
    const auto writer = CreateKernel(program, TT_RQM_H2B_H2A_WRITER_PATH, core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile});
    const auto compute = CreateKernel(
        program,
        TT_RQM_H2B_H2A_COMPUTE_PATH,
        core,
        compute_config({0, 1, 2, 3, 4, 5}));
    const uint32_t work_items = steps * component_tiles;
    SetRuntimeArgs(program, reader, core_coord, {input->address(), work_items, component_tiles});
    SetRuntimeArgs(program, compute, core_coord, {work_items});
    SetRuntimeArgs(
        program, writer, core_coord, {intermediate->address(), work_items, component_tiles});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    return {std::move(workload), 1};
}

std::vector<Assignment> assignments_for(
    const CoreRangeSet& group_1,
    const CoreRangeSet& group_2,
    uint32_t tiles_1,
    uint32_t tiles_2,
    uint32_t component_tiles) {
    std::vector<Assignment> assignments;
    uint32_t start = 0;
    for (const auto& [group, per_core] :
         {std::pair{group_1, tiles_1}, std::pair{group_2, tiles_2}}) {
        for (const auto& range : group.ranges()) {
            for (const CoreCoord& core : range) {
                assignments.push_back({core, per_core, start});
                start += per_core;
            }
        }
    }
    if (start != component_tiles) throw std::runtime_error("H1 work split is incomplete");
    return assignments;
}

PreparedProgram build_h1_program(
    const std::shared_ptr<distributed::MeshDevice>& device,
    const std::shared_ptr<distributed::MeshBuffer>& intermediate,
    const std::shared_ptr<distributed::MeshBuffer>& final_output,
    uint32_t component_tiles,
    uint32_t steps) {
    Program program = CreateProgram();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const auto [core_count, all, group_1, group_2, tiles_1, tiles_2] =
        split_work_to_cores(grid, component_tiles, true);
    for (uint32_t cb = 0; cb < 6; ++cb) create_cb(program, all, cb, 2);
    for (uint32_t cb = 6; cb < 12; ++cb) create_cb(program, all, cb);
    for (uint32_t cb = 16; cb < 22; ++cb) create_cb(program, all, cb);
    for (uint32_t cb = 24; cb < 30; ++cb) create_cb(program, all, cb);
    std::vector<uint32_t> reader_compile;
    TensorAccessorArgs(*intermediate).append_to(reader_compile);
    std::vector<uint32_t> writer_compile;
    TensorAccessorArgs(*final_output).append_to(writer_compile);
    const auto reader = CreateKernel(program, TT_RQM_H2B_H1_READER_PATH, all, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile});
    const auto writer = CreateKernel(program, TT_RQM_H2B_H1_WRITER_PATH, all, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile});
    const auto compute = CreateKernel(
        program,
        TT_RQM_H2B_H1_COMPUTE_PATH,
        all,
        compute_config({0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29}));
    for (const auto& item : assignments_for(
             group_1, group_2, tiles_1, tiles_2, component_tiles)) {
        SetRuntimeArgs(
            program,
            reader,
            item.core,
            {intermediate->address(), item.tiles, item.start, component_tiles, steps});
        SetRuntimeArgs(program, compute, item.core, {item.tiles, steps});
        SetRuntimeArgs(
            program,
            writer,
            item.core,
            {final_output->address(), item.tiles, item.start, component_tiles, 6});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    return {std::move(workload), core_count};
}

}  // namespace

int main() {
    try {
        const char* work_value = std::getenv("TT_RQM_H2B_DIR");
        const char* manifest_value = std::getenv("TT_RQM_H2B_MANIFEST");
        if (work_value == nullptr || manifest_value == nullptr) {
            throw std::runtime_error("TT_RQM_H2B_DIR and TT_RQM_H2B_MANIFEST are required");
        }
        const std::filesystem::path work_dir(work_value);
        const json manifest = json::parse(read_text(manifest_value));
        if (manifest.value("schema", "") != kProtocol ||
            manifest.value("benchmark", "") != "HamiltonianEvolutionBench" ||
            manifest.value("stage", "") != "conformance" ||
            manifest.value("dtype", "") != "float32") {
            throw std::runtime_error("unsupported H2B external manifest");
        }
        const auto shape = manifest.at("hamiltonian_shape").get<std::vector<uint32_t>>();
        if (shape.size() != 3 || shape[2] != 4 || shape[0] == 0 || shape[1] == 0) {
            throw std::runtime_error("invalid Hamiltonian shape");
        }
        const uint32_t batch = shape[0];
        const uint32_t steps = shape[1];
        const auto dt_shape = manifest.at("dt_shape").get<std::vector<uint32_t>>();
        if (!(dt_shape.empty() || dt_shape == std::vector<uint32_t>{batch, steps})) {
            throw std::runtime_error("dt must be scalar or exactly [B,K]");
        }
        const float hbar = manifest.at("hbar").get<float>();
        if (!(hbar > 0.0f) || !std::isfinite(hbar)) throw std::runtime_error("invalid hbar");
        const auto& inputs = manifest.at("inputs");
        const auto& outputs = manifest.at("outputs");
        const auto coefficients = read_words(
            work_dir / inputs.at("hamiltonians").get<std::string>(),
            static_cast<size_t>(batch) * steps * 4);
        const auto dt = read_words(
            work_dir / inputs.at("dt").get<std::string>(),
            dt_shape.empty() ? 1 : static_cast<size_t>(batch) * steps);
        const uint32_t component_tiles = (batch + kTileElements - 1) / kTileElements;
        const auto packed_input =
            pack_h2b_input(coefficients, dt, batch, steps, component_tiles, 1.0f / hbar);
        const uint32_t intermediate_bytes = steps * kLanes * component_tiles * kTileBytes;
        const uint32_t final_bytes = kLanes * component_tiles * kTileBytes;

        const auto process_start = Clock::now();
        const auto create_start = Clock::now();
        auto device = distributed::MeshDevice::create_unit_mesh(0);
        const double create_s = elapsed(create_start);
        auto& queue = device->mesh_command_queue();
        distributed::DeviceLocalBufferConfig local{
            .page_size = kTileBytes, .buffer_type = BufferType::DRAM};
        const auto allocation_start = Clock::now();
        auto input = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = intermediate_bytes}, local, device.get());
        auto intermediate = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = intermediate_bytes}, local, device.get());
        auto final_output = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = final_bytes}, local, device.get());
        const double allocation_s = elapsed(allocation_start);
        const auto build_start = Clock::now();
        auto h2a = build_h2a_program(device, input, intermediate, component_tiles, steps);
        auto h1 = build_h1_program(device, intermediate, final_output, component_tiles, steps);
        const double build_s = elapsed(build_start);

        const auto h2d_start = Clock::now();
        distributed::EnqueueWriteMeshBuffer(queue, input, packed_input, false);
        distributed::Finish(queue);
        const double h2d_s = elapsed(h2d_start);
        const auto execute_start = Clock::now();
        distributed::EnqueueMeshWorkload(queue, h2a.workload, false);
        distributed::EnqueueMeshWorkload(queue, h1.workload, false);
        distributed::Finish(queue);
        const double execute_s = elapsed(execute_start);
        const auto d2h_start = Clock::now();
        std::vector<uint32_t> packed_final;
        distributed::EnqueueReadMeshBuffer(queue, packed_final, final_output, true);
        distributed::Finish(queue);
        const double d2h_s = elapsed(d2h_start);
        const auto [final_rotors, final_phases] =
            unpack_final_output(packed_final, batch, component_tiles);
        write_words(work_dir / outputs.at("final_rotors").get<std::string>(), final_rotors);
        write_words(work_dir / outputs.at("final_phases").get<std::string>(), final_phases);
        const auto close_start = Clock::now();
        if (!device->close()) throw std::runtime_error("failed to close MeshDevice");
        const double close_s = elapsed(close_start);

        const json metadata = {
            {"implementation_class", "two_program_device_resident_h2b"},
            {"candidate_sha256", env_required("TT_RQM_H2B_CANDIDATE_SHA256")},
            {"source_commit", env_required("TT_RQM_H2B_SOURCE_COMMIT")},
            {"source_tree_clean", env_bool("TT_RQM_H2B_SOURCE_TREE_CLEAN")},
            {"source_bundle_sha256", env_required("TT_RQM_H2B_SOURCE_BUNDLE_SHA256")},
            {"tt_metal_commit", env_required("TT_RQM_H2B_TT_METAL_COMMIT")},
            {"compiler_version", env_required("TT_RQM_H2B_COMPILER_VERSION")},
            {"runtime_version", env_required("TT_RQM_H2B_RUNTIME_VERSION")},
            {"device_arch", "wormhole_b0"},
            {"device_count", 1}, {"device_id", 0},
            {"device_create_count", 1}, {"device_close_count", 1},
            {"program_count", 2}, {"h2a_core_count", h2a.core_count},
            {"h1_core_count", h1.core_count},
            {"input_layout", "step-major component-planar FP32 tiles: (step*6+lane)*component_tiles+batch_tile; h0,hx,hy,hz,dt,inverse_hbar"},
            {"intermediate_layout", "step-major component-planar FP32 tiles: (step*6+lane)*component_tiles+batch_tile; w,x,y,z,phase_real,phase_imag"},
            {"output_layout", "component-planar FP32 final tiles restored to row-major [B,4] and [B,2]"},
            {"intermediate_storage", "device_dram"},
            {"device_resident_intermediate", true},
            {"intermediate_d2h_count", 0}, {"intermediate_h2d_count", 0},
            {"host_round_trip_count", 0},
            {"h2a_arithmetic_path", "compensated H2A Dekker TwoProduct and split-2pi Tensix SFPU"},
            {"h1_arithmetic_path", "protected fused H1 Tensix compute/SFPU with L1 accumulators"},
            {"composition_order", "K-1 ... 0"},
            {"automatic_normalization", false},
        };
        const json metrics = {
            {"schema", kMetrics}, {"protocol", kProtocol},
            {"benchmark", "HamiltonianEvolutionBench"},
            {"stage", "conformance"}, {"dtype", "float32"},
            {"execution_label", "hardware"},
            {"hamiltonian_shape", shape}, {"dt_shape", dt_shape},
            {"final_rotor_shape", std::vector<uint32_t>{batch, 4}},
            {"final_phase_shape", std::vector<uint32_t>{batch, 2}},
            {"final_rotor_lane_order", {"w", "x", "y", "z"}},
            {"final_phase_lane_order", {"real", "imag"}},
            {"stable_benchmark", false}, {"performance_eligible", false},
            {"claim_level", nullptr},
            {"timings_s", {
                {"device_create", create_s}, {"buffer_allocation", allocation_s},
                {"program_build", build_s}, {"h2d", h2d_s},
                {"device_execute", execute_s}, {"d2h", d2h_s},
                {"device_close", close_s}, {"candidate_process", elapsed(process_start)}}},
            {"candidate_metadata", metadata},
        };
        write_text(
            work_dir / outputs.at("metrics").get<std::string>(), metrics.dump(2) + "\n");
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "tt_rqm_metalium_hamiltonian_evolution_candidate failed: "
                  << exc.what() << "\n";
        return 2;
    }
}
