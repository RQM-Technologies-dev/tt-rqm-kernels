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
#include <iomanip>
#include <iostream>
#include <memory>
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

constexpr std::string_view kProtocol = "tt-rqm-external-su2-compose-persistent.v1";
constexpr std::string_view kMetrics = "tt-rqm-external-su2-compose-persistent-metrics.v1";
constexpr std::string_view kImplementation = "fused_tensix_sfpu_su2_compose";
constexpr uint32_t kLanes = 6;
constexpr uint32_t kTileBytes = 32 * 32 * sizeof(uint32_t);
constexpr uint32_t kElementsPerTile = 32 * 32;
constexpr bool kPerformanceEligible = true;

struct Assignment {
    CoreCoord core;
    uint32_t tiles;
    uint32_t start;
};

struct Metadata {
    uint32_t cores;
    uint32_t component_tiles;
    uint32_t grid_x;
    uint32_t grid_y;
};

struct FusedWorkload {
    distributed::MeshWorkload workload;
    Metadata metadata;
};

struct UnfusedWorkload {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range;
    KernelHandle reader;
    KernelHandle writer;
    std::vector<Assignment> assignments;
    Metadata metadata;
};

double seconds_since(Clock::time_point start) {
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
    if (std::filesystem::file_size(path) != count * sizeof(uint32_t)) {
        throw std::runtime_error(path.filename().string() + " has an unexpected byte count");
    }
    std::vector<uint32_t> words(count);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(count * sizeof(uint32_t)));
    if (!input) throw std::runtime_error("short read from " + path.string());
    return words;
}

void write_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!output) throw std::runtime_error("failed to write " + path.string());
}

std::string fnv1a64(const std::vector<uint32_t>& words) {
    uint64_t hash = 14695981039346656037ULL;
    for (uint32_t word : words) {
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(word >> shift);
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

std::string env_or_unknown(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? "unknown" : value;
}

std::vector<uint32_t> pack_steps(
    const std::vector<uint32_t>& rotors,
    const std::vector<uint32_t>& phases,
    uint32_t batch,
    uint32_t steps) {
    const uint32_t component_tiles = (batch + kElementsPerTile - 1) / kElementsPerTile;
    const uint32_t padded = component_tiles * kElementsPerTile;
    std::vector<uint32_t> packed(static_cast<size_t>(steps) * kLanes * padded, 0);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t step = 0; step < steps; ++step) {
            for (uint32_t lane = 0; lane < 4; ++lane) {
                packed[(static_cast<size_t>(step) * kLanes + lane) * padded + b] =
                    rotors[(static_cast<size_t>(b) * steps + step) * 4 + lane];
            }
            for (uint32_t lane = 0; lane < 2; ++lane) {
                packed[(static_cast<size_t>(step) * kLanes + 4 + lane) * padded + b] =
                    phases[(static_cast<size_t>(b) * steps + step) * 2 + lane];
            }
        }
    }
    return packed;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> unpack_output(
    const std::vector<uint32_t>& packed,
    uint32_t batch) {
    const uint32_t component_tiles = (batch + kElementsPerTile - 1) / kElementsPerTile;
    const uint32_t padded = component_tiles * kElementsPerTile;
    if (packed.size() != static_cast<size_t>(kLanes) * padded) throw std::runtime_error("output size mismatch");
    std::vector<uint32_t> rotors(static_cast<size_t>(batch) * 4);
    std::vector<uint32_t> phases(static_cast<size_t>(batch) * 2);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t lane = 0; lane < 4; ++lane) rotors[static_cast<size_t>(b) * 4 + lane] = packed[lane * padded + b];
        for (uint32_t lane = 0; lane < 2; ++lane) phases[static_cast<size_t>(b) * 2 + lane] = packed[(4 + lane) * padded + b];
    }
    return {std::move(rotors), std::move(phases)};
}

void create_cb(Program& program, const CoreRangeSet& cores, uint32_t index, uint32_t depth) {
    auto cb = static_cast<tt::CBIndex>(index);
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(depth * kTileBytes, {{cb, DataFormat::Float32}}).set_page_size(cb, kTileBytes));
}

std::vector<Assignment> assignments_for(
    const CoreRangeSet& group_1,
    const CoreRangeSet& group_2,
    uint32_t tiles_1,
    uint32_t tiles_2,
    uint32_t component_tiles) {
    std::vector<Assignment> assignments;
    uint32_t start = 0;
    for (const auto& [group, per_core] : {std::pair{group_1, tiles_1}, std::pair{group_2, tiles_2}}) {
        for (const auto& range : group.ranges()) {
            for (const CoreCoord& core : range) {
                assignments.push_back({core, per_core, start});
                start += per_core;
            }
        }
    }
    if (start != component_tiles) throw std::runtime_error("work split did not cover all tiles");
    return assignments;
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

FusedWorkload build_fused(
    const std::shared_ptr<distributed::MeshDevice>& device,
    const std::shared_ptr<distributed::MeshBuffer>& input,
    const std::shared_ptr<distributed::MeshBuffer>& output,
    uint32_t component_tiles,
    uint32_t steps) {
    Program program = CreateProgram();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const auto [cores, all, group_1, group_2, tiles_1, tiles_2] = split_work_to_cores(grid, component_tiles, true);
    for (uint32_t cb = 0; cb < 6; ++cb) create_cb(program, all, cb, 2);
    for (uint32_t cb = 6; cb < 12; ++cb) create_cb(program, all, cb, 1);
    for (uint32_t cb = 16; cb < 22; ++cb) create_cb(program, all, cb, 1);
    for (uint32_t cb = 24; cb < 30; ++cb) create_cb(program, all, cb, 1);

    std::vector<uint32_t> reader_compile;
    TensorAccessorArgs(*input).append_to(reader_compile);
    std::vector<uint32_t> writer_compile;
    TensorAccessorArgs(*output).append_to(writer_compile);
    auto reader = CreateKernel(program, TT_RQM_SU2_FUSED_READER_PATH, all, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile});
    auto writer = CreateKernel(program, TT_RQM_SU2_FUSED_WRITER_PATH, all, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile});
    auto compute = CreateKernel(
        program, TT_RQM_SU2_FUSED_COMPUTE_PATH, all,
        compute_config({0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29}));
    auto work = assignments_for(group_1, group_2, tiles_1, tiles_2, component_tiles);
    for (const auto& item : work) {
        SetRuntimeArgs(program, reader, item.core, {input->address(), item.tiles, item.start, component_tiles, steps});
        SetRuntimeArgs(program, compute, item.core, {item.tiles, steps});
        SetRuntimeArgs(program, writer, item.core, {output->address(), item.tiles, item.start, component_tiles, 6});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    return {std::move(workload), {cores, component_tiles, grid.x, grid.y}};
}

UnfusedWorkload build_unfused(
    const std::shared_ptr<distributed::MeshDevice>& device,
    const std::shared_ptr<distributed::MeshBuffer>& input,
    const std::shared_ptr<distributed::MeshBuffer>& output,
    uint32_t component_tiles) {
    Program program = CreateProgram();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const auto [cores, all, group_1, group_2, tiles_1, tiles_2] = split_work_to_cores(grid, component_tiles, true);
    for (uint32_t cb = 0; cb < 12; ++cb) create_cb(program, all, cb, 2);
    for (uint32_t cb = 16; cb < 22; ++cb) create_cb(program, all, cb, 2);
    std::vector<uint32_t> reader_compile;
    TensorAccessorArgs(*input).append_to(reader_compile);
    TensorAccessorArgs(*input).append_to(reader_compile);
    std::vector<uint32_t> writer_compile;
    TensorAccessorArgs(*output).append_to(writer_compile);
    auto reader = CreateKernel(program, TT_RQM_SU2_UNFUSED_READER_PATH, all, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile});
    auto writer = CreateKernel(program, TT_RQM_SU2_UNFUSED_WRITER_PATH, all, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile});
    auto compute = CreateKernel(
        program, TT_RQM_SU2_UNFUSED_COMPUTE_PATH, all,
        compute_config({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    auto work = assignments_for(group_1, group_2, tiles_1, tiles_2, component_tiles);
    for (const auto& item : work) {
        SetRuntimeArgs(program, reader, item.core, {input->address(), input->address(), item.tiles, item.start, component_tiles, 0, 0});
        SetRuntimeArgs(program, compute, item.core, {item.tiles});
        SetRuntimeArgs(program, writer, item.core, {output->address(), item.tiles, item.start, component_tiles});
    }
    distributed::MeshCoordinateRange range(device->shape());
    distributed::MeshWorkload workload;
    workload.add_program(range, std::move(program));
    return {std::move(workload), range, reader, writer, std::move(work), {cores, component_tiles, grid.x, grid.y}};
}

void set_unfused_dispatch(
    UnfusedWorkload& prepared,
    const std::shared_ptr<distributed::MeshBuffer>& lhs,
    const std::shared_ptr<distributed::MeshBuffer>& rhs,
    const std::shared_ptr<distributed::MeshBuffer>& output,
    uint32_t lhs_page_base,
    uint32_t rhs_page_base) {
    auto& program = prepared.workload.get_programs().at(prepared.device_range);
    for (const auto& item : prepared.assignments) {
        SetRuntimeArgs(program, prepared.reader, item.core,
            {lhs->address(), rhs->address(), item.tiles, item.start, prepared.metadata.component_tiles, lhs_page_base, rhs_page_base});
        SetRuntimeArgs(program, prepared.writer, item.core,
            {output->address(), item.tiles, item.start, prepared.metadata.component_tiles});
    }
}

void enqueue_unfused_chain(
    distributed::MeshCommandQueue& cq,
    UnfusedWorkload& prepared,
    const std::shared_ptr<distributed::MeshBuffer>& input,
    const std::shared_ptr<distributed::MeshBuffer>& ping_a,
    const std::shared_ptr<distributed::MeshBuffer>& ping_b,
    uint32_t steps) {
    const uint32_t stride = kLanes * prepared.metadata.component_tiles;
    for (uint32_t step = 1; step < steps; ++step) {
        const auto& rhs = step == 1 ? input : ((step % 2 == 0) ? ping_a : ping_b);
        const auto& output = step % 2 == 1 ? ping_a : ping_b;
        const uint32_t rhs_base = step == 1 ? 0 : 0;
        set_unfused_dispatch(prepared, input, rhs, output, step * stride, rhs_base);
        distributed::EnqueueMeshWorkload(cq, prepared.workload, true);
    }
}

json run_case(
    const std::shared_ptr<distributed::MeshDevice>& device,
    const std::filesystem::path& workdir,
    const json& spec) {
    const uint32_t batch = spec.at("B").get<uint32_t>();
    const uint32_t steps = spec.at("K").get<uint32_t>();
    const uint32_t repeats = spec.at("repeat_count").get<uint32_t>();
    const uint32_t warmups = spec.at("warmup_pairs").get<uint32_t>();
    const uint32_t samples = spec.at("samples").get<uint32_t>();
    if (batch == 0 || steps < 2 || repeats == 0 || samples == 0) throw std::runtime_error("invalid case dimensions");
    const auto& inputs = spec.at("inputs");
    const auto& outputs = spec.at("outputs");
    const auto rotors = read_words(workdir / inputs.at("rotors").get<std::string>(), static_cast<size_t>(batch) * steps * 4);
    const auto phases = read_words(workdir / inputs.at("phases").get<std::string>(), static_cast<size_t>(batch) * steps * 2);
    const auto packed = pack_steps(rotors, phases, batch, steps);
    const uint32_t component_tiles = (batch + kElementsPerTile - 1) / kElementsPerTile;
    const uint32_t input_bytes = steps * kLanes * component_tiles * kTileBytes;
    const uint32_t output_bytes = kLanes * component_tiles * kTileBytes;
    auto& cq = device->mesh_command_queue();
    distributed::DeviceLocalBufferConfig local{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};

    const auto allocation_start = Clock::now();
    distributed::ReplicatedBufferConfig input_config{.size = input_bytes};
    distributed::ReplicatedBufferConfig output_config{.size = output_bytes};
    auto input = distributed::MeshBuffer::create(input_config, local, device.get());
    auto fused_output = distributed::MeshBuffer::create(output_config, local, device.get());
    auto ping_a = distributed::MeshBuffer::create(output_config, local, device.get());
    auto ping_b = distributed::MeshBuffer::create(output_config, local, device.get());
    const double allocation_s = seconds_since(allocation_start);

    const auto build_start = Clock::now();
    auto fused = build_fused(device, input, fused_output, component_tiles, steps);
    auto unfused = build_unfused(device, input, ping_a, component_tiles);
    const double build_s = seconds_since(build_start);

    const auto h2d_start = Clock::now();
    distributed::EnqueueWriteMeshBuffer(cq, input, packed, false);
    distributed::Finish(cq);
    const double h2d_s = seconds_since(h2d_start);

    const auto run_fused = [&]() {
        const auto start = Clock::now();
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) distributed::EnqueueMeshWorkload(cq, fused.workload, true);
        distributed::Finish(cq);
        return seconds_since(start);
    };
    const auto run_unfused = [&]() {
        const auto start = Clock::now();
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) enqueue_unfused_chain(cq, unfused, input, ping_a, ping_b, steps);
        distributed::Finish(cq);
        return seconds_since(start);
    };

    const auto warmup_start = Clock::now();
    for (uint32_t warmup = 0; warmup < warmups; ++warmup) {
        if (warmup % 2 == 0) { run_fused(); run_unfused(); }
        else { run_unfused(); run_fused(); }
    }
    const double warmup_s = seconds_since(warmup_start);

    std::vector<double> fused_samples;
    std::vector<double> unfused_samples;
    std::vector<std::string> order;
    for (uint32_t sample = 0; sample < samples; ++sample) {
        if (sample % 2 == 0) {
            order.push_back("fused_first");
            fused_samples.push_back(run_fused());
            unfused_samples.push_back(run_unfused());
        } else {
            order.push_back("unfused_first");
            unfused_samples.push_back(run_unfused());
            fused_samples.push_back(run_fused());
        }
    }

    const auto d2h_start = Clock::now();
    std::vector<uint32_t> fused_packed;
    std::vector<uint32_t> unfused_packed;
    distributed::EnqueueReadMeshBuffer(cq, fused_packed, fused_output, true);
    auto final_unfused = ((steps - 1) % 2 == 1) ? ping_a : ping_b;
    distributed::EnqueueReadMeshBuffer(cq, unfused_packed, final_unfused, true);
    distributed::Finish(cq);
    const double d2h_s = seconds_since(d2h_start);
    auto [fused_rotors, fused_phases] = unpack_output(fused_packed, batch);
    auto [unfused_rotors, unfused_phases] = unpack_output(unfused_packed, batch);
    write_words(workdir / outputs.at("fused_rotors").get<std::string>(), fused_rotors);
    write_words(workdir / outputs.at("fused_phases").get<std::string>(), fused_phases);
    write_words(workdir / outputs.at("unfused_rotors").get<std::string>(), unfused_rotors);
    write_words(workdir / outputs.at("unfused_phases").get<std::string>(), unfused_phases);

    return {
        {"case_id", spec.at("case_id")}, {"B", batch}, {"K", steps},
        {"repeat_count", repeats}, {"warmup_pairs", warmups}, {"samples", samples},
        {"input_identity", {{"rotors_sha256", inputs.at("rotors_sha256")}, {"phases_sha256", inputs.at("phases_sha256")}}},
        {"output_identity", {
            {"fused_rotors_fnv1a64", fnv1a64(fused_rotors)}, {"fused_phases_fnv1a64", fnv1a64(fused_phases)},
            {"unfused_rotors_fnv1a64", fnv1a64(unfused_rotors)}, {"unfused_phases_fnv1a64", fnv1a64(unfused_phases)},
            {"value_count_per_path", static_cast<uint64_t>(batch) * kLanes}
        }},
        {"timings_s", {{"buffer_allocation", allocation_s}, {"program_build", build_s}, {"h2d", h2d_s},
            {"warmup", warmup_s}, {"fused_samples", fused_samples}, {"unfused_samples", unfused_samples},
            {"paired_order", order}, {"d2h", d2h_s}}},
        {"work", {{"device_count", 1}, {"device_id", 0}, {"core_count", fused.metadata.cores},
            {"available_core_count", fused.metadata.grid_x * fused.metadata.grid_y},
            {"component_tiles", component_tiles}, {"layout", "step_major_planar_float32_tiles_32x32"},
            {"work_split", "row_major"}, {"fused_dispatches_per_chain", 1},
            {"unfused_dispatches_per_chain", steps - 1}, {"arithmetic_path", "tensix_compute_sfpu"},
            {"fused_accumulator_storage", "tensix_l1_ping_pong"}}}
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path workdir;
        std::filesystem::path manifest_path;
        int device_id = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (++i >= argc) throw std::runtime_error("missing argument value");
            if (arg == "--workdir") workdir = argv[i];
            else if (arg == "--manifest") manifest_path = argv[i];
            else if (arg == "--device") device_id = std::stoi(argv[i]);
            else throw std::runtime_error("unknown argument: " + std::string(arg));
        }
        if (workdir.empty()) {
            const char* value = std::getenv("TT_RQM_SU2_COMPOSE_DIR");
            if (value != nullptr) workdir = value;
        }
        if (manifest_path.empty()) manifest_path = workdir / "manifest.json";
        if (workdir.empty() || device_id != 0) throw std::runtime_error("candidate requires a workdir and Wormhole device 0");
        const json manifest = json::parse(read_text(manifest_path));
        if (manifest.value("schema", "") != kProtocol || manifest.value("workload", "") != "su2_compose" ||
            manifest.value("dtype", "") != "float32") throw std::runtime_error("unsupported SU2ComposeBench manifest");

        const auto process_start = Clock::now();
        const auto create_start = Clock::now();
        auto device = distributed::MeshDevice::create_unit_mesh(device_id);
        const double create_s = seconds_since(create_start);
        json cases = json::array();
        for (const auto& spec : manifest.at("cases")) cases.push_back(run_case(device, workdir, spec));
        const auto close_start = Clock::now();
        if (!device->close()) throw std::runtime_error("failed to close MeshDevice");
        const double close_s = seconds_since(close_start);
        const json metrics = {
            {"schema", kMetrics}, {"protocol", kProtocol}, {"backend", "tt-metalium-su2-compose-candidate"},
            {"device", "tenstorrent/wormhole-device-0"}, {"dtype", "float32"}, {"execution_kind", "hardware"},
            {"implementation_class", kImplementation}, {"performance_eligible", kPerformanceEligible},
            {"stable_benchmark", false}, {"lifecycle", {{"device_count", 1}, {"device_id", 0}, {"create_count", 1}, {"close_count", 1}}},
            {"session_timings_s", {{"device_create", create_s}, {"device_close", close_s}, {"candidate_session", seconds_since(process_start)}}},
            {"cases", cases}, {"provenance", {
                {"chip_type", env_or_unknown("TT_RQM_CHIP_TYPE")}, {"tt_metal_commit", env_or_unknown("TT_RQM_TT_METAL_COMMIT")},
                {"compiler_version", env_or_unknown("TT_RQM_COMPILER_VERSION")}, {"runtime_version", env_or_unknown("TT_RQM_RUNTIME_VERSION")},
                {"build_id", env_or_unknown("TT_RQM_BUILD_ID")}, {"candidate_sha256", env_or_unknown("TT_RQM_CANDIDATE_SHA256")},
                {"repository_commit", env_or_unknown("TT_RQM_REPOSITORY_COMMIT")},
                {"timer_scope", "one persistent device session; paired prepared enqueue-through-Finish samples"}
            }}
        };
        write_text(workdir / manifest.at("outputs").at("metrics").get<std::string>(), metrics.dump(2) + "\n");
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "tt_rqm_metalium_su2_compose_candidate failed: " << exc.what() << "\n";
        return 2;
    }
}
