// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

#ifndef TT_RQM_QMUL_READER_KERNEL_PATH
#error "TT_RQM_QMUL_READER_KERNEL_PATH is required"
#endif
#ifndef TT_RQM_QMUL_COMPUTE_KERNEL_PATH
#error "TT_RQM_QMUL_COMPUTE_KERNEL_PATH is required"
#endif
#ifndef TT_RQM_QMUL_WRITER_KERNEL_PATH
#error "TT_RQM_QMUL_WRITER_KERNEL_PATH is required"
#endif

namespace {

constexpr std::string_view kProtocol = "tt-rqm-external-qmul.v1";
constexpr std::string_view kMetricsSchema = "tt-rqm-external-qmul-metrics.v2";
constexpr uint32_t kLanes = 4;
constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kElementsPerTile = kTileWidth * kTileHeight;
constexpr uint32_t kTileBytes = kElementsPerTile * sizeof(uint32_t);
constexpr bool kPerformanceEligible = true;

struct Config {
    std::filesystem::path workdir;
    std::filesystem::path manifest_path;
    int device_id = 0;
};

struct Manifest {
    uint32_t items = 0;
    uint32_t iterations = 0;
    uint32_t warmup = 0;
};

struct WorkloadMetadata {
    uint32_t core_count = 0;
    uint32_t component_tiles = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
};

struct PreparedWorkload {
    distributed::MeshWorkload workload;
    WorkloadMetadata metadata;
};

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to read " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write " + path.string());
    }
    output << text;
}

std::filesystem::path env_path(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::filesystem::path{} : std::filesystem::path(value);
}

std::string json_string(const std::string& text, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("manifest missing key: " + std::string(key));
    }
    const auto colon_pos = text.find(':', key_pos + needle.size());
    const auto quote_pos = text.find('"', colon_pos + 1);
    const auto end_quote_pos = text.find('"', quote_pos + 1);
    if (colon_pos == std::string::npos || quote_pos == std::string::npos || end_quote_pos == std::string::npos) {
        throw std::runtime_error("manifest key is not a string: " + std::string(key));
    }
    return text.substr(quote_pos + 1, end_quote_pos - quote_pos - 1);
}

uint32_t json_uint(const std::string& text, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("manifest missing key: " + std::string(key));
    }
    const auto colon_pos = text.find(':', key_pos + needle.size());
    auto value_pos = colon_pos + 1;
    while (value_pos < text.size() && std::isspace(static_cast<unsigned char>(text[value_pos]))) {
        ++value_pos;
    }
    auto end_pos = value_pos;
    while (end_pos < text.size() && std::isdigit(static_cast<unsigned char>(text[end_pos]))) {
        ++end_pos;
    }
    if (colon_pos == std::string::npos || end_pos == value_pos) {
        throw std::runtime_error("manifest key is not an unsigned integer: " + std::string(key));
    }
    const auto parsed = std::stoull(text.substr(value_pos, end_pos - value_pos));
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("manifest key is too large: " + std::string(key));
    }
    return static_cast<uint32_t>(parsed);
}

Manifest load_manifest(const std::filesystem::path& path) {
    const std::string text = read_text(path);
    if (json_string(text, "schema") != kProtocol || json_string(text, "workload") != "qmul" ||
        json_string(text, "dtype") != "float32") {
        throw std::runtime_error("unsupported external-qmul manifest");
    }
    Manifest manifest{
        .items = json_uint(text, "items"),
        .iterations = json_uint(text, "iterations"),
        .warmup = json_uint(text, "warmup"),
    };
    if (manifest.items == 0 || manifest.iterations == 0) {
        throw std::runtime_error("items and iterations must be positive");
    }
    return manifest;
}

std::vector<uint32_t> read_words(const std::filesystem::path& path, uint32_t items) {
    const auto expected_bytes = static_cast<std::uintmax_t>(items) * kLanes * sizeof(uint32_t);
    if (std::filesystem::file_size(path) != expected_bytes) {
        throw std::runtime_error(path.filename().string() + " has an unexpected byte count");
    }
    std::vector<uint32_t> words(static_cast<size_t>(items) * kLanes);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(expected_bytes));
    if (!input) {
        throw std::runtime_error("short read from " + path.string());
    }
    return words;
}

void write_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!output) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

std::vector<uint32_t> aos_to_planar_tiles(const std::vector<uint32_t>& aos, uint32_t items) {
    const uint32_t component_tiles = (items + kElementsPerTile - 1) / kElementsPerTile;
    const uint32_t padded_items = component_tiles * kElementsPerTile;
    std::vector<uint32_t> planar(static_cast<size_t>(kLanes) * padded_items, 0);
    for (uint32_t item = 0; item < items; ++item) {
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            planar[static_cast<size_t>(lane) * padded_items + item] = aos[static_cast<size_t>(item) * kLanes + lane];
        }
    }
    return planar;
}

std::vector<uint32_t> planar_tiles_to_aos(const std::vector<uint32_t>& planar, uint32_t items) {
    const uint32_t component_tiles = (items + kElementsPerTile - 1) / kElementsPerTile;
    const uint32_t padded_items = component_tiles * kElementsPerTile;
    if (planar.size() != static_cast<size_t>(kLanes) * padded_items) {
        throw std::runtime_error("planar output size mismatch");
    }
    std::vector<uint32_t> aos(static_cast<size_t>(items) * kLanes);
    for (uint32_t item = 0; item < items; ++item) {
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            aos[static_cast<size_t>(item) * kLanes + lane] = planar[static_cast<size_t>(lane) * padded_items + item];
        }
    }
    return aos;
}

Config parse_config(int argc, char** argv) {
    Config config{.workdir = env_path("TT_RQM_EXTERNAL_QMUL_DIR"), .manifest_path = env_path("TT_RQM_EXTERNAL_QMUL_MANIFEST")};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&]() -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + std::string(arg));
            }
            return argv[++i];
        };
        if (arg == "--workdir") {
            config.workdir = require_value();
        } else if (arg == "--manifest") {
            config.manifest_path = require_value();
        } else if (arg == "--device") {
            config.device_id = std::stoi(require_value());
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--workdir DIR] [--manifest FILE] [--device 0]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (config.workdir.empty()) {
        throw std::runtime_error("TT_RQM_EXTERNAL_QMUL_DIR is required unless --workdir is set");
    }
    if (config.manifest_path.empty()) {
        config.manifest_path = config.workdir / "manifest.json";
    }
    if (config.device_id != 0) {
        throw std::runtime_error("Stage B candidate is restricted to Wormhole device 0");
    }
    return config;
}

void create_float32_cb(Program& program, const CoreRangeSet& cores, tt::CBIndex cb) {
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(2 * kTileBytes, {{cb, DataFormat::Float32}}).set_page_size(cb, kTileBytes));
}

PreparedWorkload build_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::shared_ptr<distributed::MeshBuffer>& a,
    const std::shared_ptr<distributed::MeshBuffer>& b,
    const std::shared_ptr<distributed::MeshBuffer>& out,
    uint32_t component_tiles) {
    Program program = CreateProgram();
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const auto [core_count, all_cores, group_1, group_2, tiles_1, tiles_2] =
        split_work_to_cores(grid, component_tiles, true);

    for (uint32_t lane = 0; lane < 8; ++lane) {
        create_float32_cb(program, all_cores, static_cast<tt::CBIndex>(static_cast<uint32_t>(tt::CBIndex::c_0) + lane));
    }
    for (uint32_t lane = 0; lane < 4; ++lane) {
        create_float32_cb(program, all_cores, static_cast<tt::CBIndex>(static_cast<uint32_t>(tt::CBIndex::c_16) + lane));
    }

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a).append_to(reader_args);
    TensorAccessorArgs(*b).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*out).append_to(writer_args);

    auto reader = CreateKernel(
        program,
        TT_RQM_QMUL_READER_KERNEL_PATH,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args});
    auto writer = CreateKernel(
        program,
        TT_RQM_QMUL_WRITER_KERNEL_PATH,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args});

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t lane = 0; lane < 8; ++lane) {
        unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_0) + lane] = UnpackToDestMode::UnpackToDestFp32;
    }
    auto compute = CreateKernel(
        program,
        TT_RQM_QMUL_COMPUTE_KERNEL_PATH,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode = false,
        });

    uint32_t start_tile = 0;
    for (const auto& [group, tiles_per_core] : {std::pair{group_1, tiles_1}, std::pair{group_2, tiles_2}}) {
        for (const auto& range : group.ranges()) {
            for (const CoreCoord& core : range) {
                SetRuntimeArgs(program, reader, core, {a->address(), b->address(), tiles_per_core, start_tile, component_tiles});
                SetRuntimeArgs(program, compute, core, {tiles_per_core});
                SetRuntimeArgs(program, writer, core, {out->address(), tiles_per_core, start_tile, component_tiles});
                start_tile += tiles_per_core;
            }
        }
    }
    if (start_tile != component_tiles) {
        throw std::runtime_error("work split did not cover every component tile");
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    return {
        .workload = std::move(workload),
        .metadata = {
            .core_count = core_count,
            .component_tiles = component_tiles,
            .grid_x = grid.x,
            .grid_y = grid.y,
        },
    };
}

std::vector<uint32_t> run_candidate(
    const std::vector<uint32_t>& a_aos,
    const std::vector<uint32_t>& b_aos,
    const Manifest& manifest,
    int device_id,
    double& setup_s,
    double& device_s,
    WorkloadMetadata& metadata) {
    const auto setup_start = std::chrono::steady_clock::now();
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    auto& cq = mesh_device->mesh_command_queue();
    const uint32_t component_tiles = (manifest.items + kElementsPerTile - 1) / kElementsPerTile;
    const uint32_t total_tiles = kLanes * component_tiles;
    const uint32_t total_bytes = total_tiles * kTileBytes;

    distributed::DeviceLocalBufferConfig local_config{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_bytes};
    auto a = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    auto b = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    auto out = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    const auto a_planar = aos_to_planar_tiles(a_aos, manifest.items);
    const auto b_planar = aos_to_planar_tiles(b_aos, manifest.items);
    distributed::EnqueueWriteMeshBuffer(cq, a, a_planar, false);
    distributed::EnqueueWriteMeshBuffer(cq, b, b_planar, false);
    auto prepared = build_workload(mesh_device, a, b, out, component_tiles);
    metadata = prepared.metadata;
    distributed::Finish(cq);
    const auto setup_end = std::chrono::steady_clock::now();
    setup_s = std::chrono::duration<double>(setup_end - setup_start).count();

    for (uint32_t i = 0; i < manifest.warmup; ++i) {
        distributed::EnqueueMeshWorkload(cq, prepared.workload, true);
    }
    const auto device_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < manifest.iterations; ++i) {
        distributed::EnqueueMeshWorkload(cq, prepared.workload, true);
    }
    distributed::Finish(cq);
    const auto device_end = std::chrono::steady_clock::now();
    device_s = std::chrono::duration<double>(device_end - device_start).count();

    std::vector<uint32_t> out_planar;
    distributed::EnqueueReadMeshBuffer(cq, out_planar, out, true);
    if (!mesh_device->close()) {
        throw std::runtime_error("failed to close MeshDevice cleanly");
    }
    return planar_tiles_to_aos(out_planar, manifest.items);
}

std::string execution_kind() {
    const char* configured = std::getenv("TT_RQM_EXECUTION_LABEL");
    return configured == nullptr ? "hardware" : configured;
}

std::string env_or_unknown(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || std::string_view(value).empty() ? "unknown" : value;
}

void write_metrics(
    const std::filesystem::path& path,
    const Manifest& manifest,
    double setup_s,
    double device_s,
    const WorkloadMetadata& metadata) {
    std::ostringstream metrics;
    metrics << "{\n"
            << "  \"schema\": \"" << kMetricsSchema << "\",\n"
            << "  \"protocol\": \"" << kProtocol << "\",\n"
            << "  \"backend\": \"tt-metalium-qmul-multicore-candidate\",\n"
            << "  \"device\": \"tenstorrent/wormhole-device-0\",\n"
            << "  \"dtype\": \"float32\",\n"
            << "  \"items\": " << manifest.items << ",\n"
            << "  \"iterations\": " << manifest.iterations << ",\n"
            << "  \"warmup\": " << manifest.warmup << ",\n"
            << "  \"execution_kind\": \"" << execution_kind() << "\",\n"
            << "  \"implementation_class\": \"multicore_tensix_sfpu_qmul\",\n"
            << "  \"performance_eligible\": " << (kPerformanceEligible ? "true" : "false") << ",\n"
            << "  \"timings_s\": {\"setup\": " << setup_s << ", \"device\": " << device_s << "},\n"
            << "  \"work\": {\"device_count\": 1, \"device_id\": 0, \"core_count\": " << metadata.core_count
            << ", \"component_tiles\": " << metadata.component_tiles
            << ", \"grid_x\": " << metadata.grid_x << ", \"grid_y\": " << metadata.grid_y
            << ", \"available_core_count\": " << metadata.grid_x * metadata.grid_y
            << ", \"layout\": \"planar_float32_tiles_32x32\", \"work_split\": \"row_major\", "
               "\"arithmetic_path\": \"tensix_compute_sfpu\"},\n"
            << "  \"provenance\": {\n"
            << "    \"chip_type\": \"" << env_or_unknown("TT_RQM_CHIP_TYPE") << "\",\n"
            << "    \"tt_metal_commit\": \"" << env_or_unknown("TT_RQM_TT_METAL_COMMIT") << "\",\n"
            << "    \"compiler_version\": \"" << env_or_unknown("TT_RQM_COMPILER_VERSION") << "\",\n"
            << "    \"runtime_version\": \"" << env_or_unknown("TT_RQM_RUNTIME_VERSION") << "\",\n"
            << "    \"build_id\": \"" << env_or_unknown("TT_RQM_BUILD_ID") << "\",\n"
            << "    \"timer_scope\": \"prepared multicore workload enqueue plus Finish; excludes setup and readback\"\n"
            << "  }\n"
            << "}\n";
    write_text(path, metrics.str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_config(argc, argv);
        const Manifest manifest = load_manifest(config.manifest_path);
        const auto a = read_words(config.workdir / "a.bin", manifest.items);
        const auto b = read_words(config.workdir / "b.bin", manifest.items);
        double setup_s = 0.0;
        double device_s = 0.0;
        WorkloadMetadata metadata;
        const auto out = run_candidate(a, b, manifest, config.device_id, setup_s, device_s, metadata);
        write_words(config.workdir / "out.bin", out);
        write_metrics(config.workdir / "metrics.json", manifest, setup_s, device_s, metadata);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "tt_rqm_metalium_qmul_multicore_candidate failed: " << exc.what() << "\n";
        return 2;
    }
}
