// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

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

#ifndef TT_RQM_QMUL_KERNEL_PATH
#error "TT_RQM_QMUL_KERNEL_PATH must point to kernels/qmul_riscv.cpp"
#endif

namespace {

constexpr std::string_view kProtocol = "tt-rqm-external-qmul.v1";
constexpr std::string_view kMetricsSchema = "tt-rqm-external-qmul-metrics.v2";
constexpr uint32_t kLanes = 4;
constexpr uint32_t kQuaternionBytes = kLanes * sizeof(uint32_t);

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
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("manifest key has no value: " + std::string(key));
    }
    const auto quote_pos = text.find('"', colon_pos + 1);
    const auto end_quote_pos = text.find('"', quote_pos + 1);
    if (quote_pos == std::string::npos || end_quote_pos == std::string::npos) {
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
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("manifest key has no value: " + std::string(key));
    }
    auto value_pos = colon_pos + 1;
    while (value_pos < text.size() && std::isspace(static_cast<unsigned char>(text[value_pos]))) {
        ++value_pos;
    }
    auto end_pos = value_pos;
    while (end_pos < text.size() && std::isdigit(static_cast<unsigned char>(text[end_pos]))) {
        ++end_pos;
    }
    if (end_pos == value_pos) {
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
    if (json_string(text, "schema") != kProtocol) {
        throw std::runtime_error("unsupported external-qmul protocol");
    }
    if (json_string(text, "workload") != "qmul") {
        throw std::runtime_error("unsupported workload");
    }
    if (json_string(text, "dtype") != "float32") {
        throw std::runtime_error("unsupported dtype");
    }
    Manifest manifest{
        .items = json_uint(text, "items"),
        .iterations = json_uint(text, "iterations"),
        .warmup = json_uint(text, "warmup"),
    };
    if (manifest.items == 0) {
        throw std::runtime_error("items must be positive");
    }
    if (manifest.iterations == 0) {
        throw std::runtime_error("iterations must be positive");
    }
    return manifest;
}

std::vector<uint32_t> read_words(const std::filesystem::path& path, uint32_t items) {
    const auto expected_bytes = static_cast<std::uintmax_t>(items) * kQuaternionBytes;
    const auto actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error(
            path.filename().string() + " has " + std::to_string(actual_bytes) + " bytes; expected " +
            std::to_string(expected_bytes));
    }

    std::vector<uint32_t> words(static_cast<size_t>(items) * kLanes);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to read " + path.string());
    }
    input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(expected_bytes));
    if (!input) {
        throw std::runtime_error("short read from " + path.string());
    }
    return words;
}

void write_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to write " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(words.data()),
        static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!output) {
        throw std::runtime_error("short write to " + path.string());
    }
}

Config parse_config(int argc, char** argv) {
    Config config;
    config.workdir = env_path("TT_RQM_EXTERNAL_QMUL_DIR");
    config.manifest_path = env_path("TT_RQM_EXTERNAL_QMUL_MANIFEST");

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](std::string_view name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + std::string(name));
            }
            return argv[++i];
        };
        if (arg == "--workdir") {
            config.workdir = require_value(arg);
        } else if (arg == "--manifest") {
            config.manifest_path = require_value(arg);
        } else if (arg == "--device") {
            config.device_id = std::stoi(require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--workdir DIR] [--manifest FILE] [--device ID]\n";
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
    return config;
}

distributed::MeshWorkload build_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::shared_ptr<distributed::MeshBuffer>& a_dram_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& b_dram_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& out_dram_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& a_l1_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& b_l1_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& out_l1_buffer,
    uint32_t items) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    KernelHandle kernel = CreateKernel(
        program,
        TT_RQM_QMUL_KERNEL_PATH,
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(
        program,
        kernel,
        core,
        {
            a_dram_buffer->address(),
            b_dram_buffer->address(),
            out_dram_buffer->address(),
            a_l1_buffer->address(),
            b_l1_buffer->address(),
            out_l1_buffer->address(),
            items,
        });

    workload.add_program(device_range, std::move(program));
    return workload;
}

std::vector<uint32_t> run_qmul_candidate(
    const std::vector<uint32_t>& a_words,
    const std::vector<uint32_t>& b_words,
    const Manifest& manifest,
    int device_id,
    double& setup_s,
    double& device_s) {
    const auto setup_start = std::chrono::steady_clock::now();
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    const uint32_t total_bytes = manifest.items * kQuaternionBytes;
    distributed::DeviceLocalBufferConfig dram_config{.page_size = kQuaternionBytes, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1_config{.page_size = kQuaternionBytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig dram_buffer_config{.size = total_bytes};
    distributed::ReplicatedBufferConfig l1_buffer_config{.size = kQuaternionBytes};

    auto a_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto b_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto out_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto a_l1_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());
    auto b_l1_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());
    auto out_l1_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());

    distributed::EnqueueWriteMeshBuffer(cq, a_dram_buffer, a_words, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram_buffer, b_words, false);

    auto workload = build_workload(
        mesh_device,
        a_dram_buffer,
        b_dram_buffer,
        out_dram_buffer,
        a_l1_buffer,
        b_l1_buffer,
        out_l1_buffer,
        manifest.items);
    distributed::Finish(cq);
    const auto setup_end = std::chrono::steady_clock::now();
    setup_s = std::chrono::duration<double>(setup_end - setup_start).count();

    for (uint32_t i = 0; i < manifest.warmup; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, true);
    }

    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < manifest.iterations; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, true);
    }
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();
    device_s = std::chrono::duration<double>(end - start).count();

    std::vector<uint32_t> out_words;
    distributed::EnqueueReadMeshBuffer(cq, out_words, out_dram_buffer, true);
    if (!mesh_device->close()) {
        throw std::runtime_error("failed to close MeshDevice cleanly");
    }
    return out_words;
}

std::string device_label() {
    if (std::getenv("TT_METAL_EMULE_MODE") != nullptr) {
        return "tt-emule/tt-metalium-riscv-qmul-candidate";
    }
    return "tt-metalium-riscv-qmul-candidate";
}

std::string execution_kind() {
    const char* configured = std::getenv("TT_RQM_EXECUTION_LABEL");
    if (configured != nullptr) {
        return configured;
    }
    return std::getenv("TT_METAL_EMULE_MODE") != nullptr ? "emulation" : "hardware";
}

std::string env_or_unknown(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || std::string_view(value).empty() ? "unknown" : value;
}

void write_metrics(
    const std::filesystem::path& path,
    const Manifest& manifest,
    double setup_s,
    double device_s) {
    std::ostringstream metrics;
    metrics << "{\n"
            << "  \"schema\": \"" << kMetricsSchema << "\",\n"
            << "  \"protocol\": \"" << kProtocol << "\",\n"
            << "  \"backend\": \"tt-metalium-qmul-riscv-candidate\",\n"
            << "  \"device\": \"" << device_label() << "\",\n"
            << "  \"dtype\": \"float32\",\n"
            << "  \"items\": " << manifest.items << ",\n"
            << "  \"iterations\": " << manifest.iterations << ",\n"
            << "  \"warmup\": " << manifest.warmup << ",\n"
            << "  \"execution_kind\": \"" << execution_kind() << "\",\n"
            << "  \"implementation_class\": \"scalar_riscv_correctness_baseline\",\n"
            << "  \"performance_eligible\": false,\n"
            << "  \"timings_s\": {\"setup\": " << setup_s
            << ", \"device\": " << device_s << "},\n"
            << "  \"provenance\": {\n"
            << "    \"chip_type\": \"" << env_or_unknown("TT_RQM_CHIP_TYPE") << "\",\n"
            << "    \"tt_metal_commit\": \"" << env_or_unknown("TT_RQM_TT_METAL_COMMIT") << "\",\n"
            << "    \"compiler_version\": \"" << env_or_unknown("TT_RQM_COMPILER_VERSION") << "\",\n"
            << "    \"runtime_version\": \"" << env_or_unknown("TT_RQM_RUNTIME_VERSION") << "\",\n"
            << "    \"build_id\": \"" << env_or_unknown("TT_RQM_BUILD_ID") << "\",\n"
            << "    \"timer_scope\": \"prepared workload enqueue plus Finish; excludes setup and readback\"\n"
            << "  },\n"
            << "  \"note\": \"Experimental TT-Metalium RISC-V qmul candidate; validate execution label before using as emulation or hardware evidence.\"\n"
            << "}\n";
    write_text(path, metrics.str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config config = parse_config(argc, argv);
        Manifest manifest = load_manifest(config.manifest_path);
        const auto a_words = read_words(config.workdir / "a.bin", manifest.items);
        const auto b_words = read_words(config.workdir / "b.bin", manifest.items);

        double setup_s = 0.0;
        double device_s = 0.0;
        auto out_words = run_qmul_candidate(
            a_words, b_words, manifest, config.device_id, setup_s, device_s);
        if (out_words.size() != static_cast<size_t>(manifest.items) * kLanes) {
            throw std::runtime_error("candidate output size mismatch");
        }
        write_words(config.workdir / "out.bin", out_words);
        write_metrics(config.workdir / "metrics.json", manifest, setup_s, device_s);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "tt_metalium_qmul_candidate failed: " << exc.what() << "\n";
        return 2;
    }
}
