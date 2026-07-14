// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>

#define main tt_rqm_embedded_single_run_main
#include "qmul_multicore_candidate.cpp"
#undef main

#include <iomanip>
#include <memory>

namespace persistent_qmul {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kPersistentProtocol = "tt-rqm-external-qmul-persistent.v1";
constexpr std::string_view kPersistentMetrics = "tt-rqm-external-qmul-persistent-metrics.v1";
constexpr std::string_view kImplementationClass = "multicore_tensix_sfpu_qmul_persistent";

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
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

class PersistentDeviceSession {
public:
    explicit PersistentDeviceSession(int device_id) {
        if (device_id != 0) {
            throw std::runtime_error("persistent Stage B candidate is restricted to Wormhole device 0");
        }
        const auto started = Clock::now();
        device_ = distributed::MeshDevice::create_unit_mesh(device_id);
        create_s_ = seconds_since(started);
        create_count_ = 1;
    }

    PersistentDeviceSession(const PersistentDeviceSession&) = delete;
    PersistentDeviceSession& operator=(const PersistentDeviceSession&) = delete;

    ~PersistentDeviceSession() {
        if (device_ && !closed_) {
            try {
                close();
            } catch (...) {
            }
        }
    }

    std::shared_ptr<distributed::MeshDevice> device() const { return device_; }
    auto& command_queue() { return device_->mesh_command_queue(); }
    double create_s() const { return create_s_; }
    uint32_t create_count() const { return create_count_; }
    uint32_t close_count() const { return close_count_; }

    double close() {
        if (closed_) {
            return 0.0;
        }
        const auto started = Clock::now();
        if (!device_->close()) {
            throw std::runtime_error("failed to close persistent MeshDevice cleanly");
        }
        closed_ = true;
        close_count_ = 1;
        return seconds_since(started);
    }

private:
    std::shared_ptr<distributed::MeshDevice> device_;
    double create_s_ = 0.0;
    uint32_t create_count_ = 0;
    uint32_t close_count_ = 0;
    bool closed_ = false;
};

struct PersistentConfig {
    std::filesystem::path workdir;
    std::filesystem::path manifest;
    int device_id = 0;
};

PersistentConfig parse_persistent_config(int argc, char** argv) {
    PersistentConfig config{
        .workdir = env_path("TT_RQM_PERSISTENT_QMUL_DIR"),
        .manifest = env_path("TT_RQM_PERSISTENT_QMUL_MANIFEST"),
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto next = [&]() -> const char* {
            if (++i >= argc) throw std::runtime_error("missing value after " + std::string(arg));
            return argv[i];
        };
        if (arg == "--workdir") config.workdir = next();
        else if (arg == "--manifest") config.manifest = next();
        else if (arg == "--device") config.device_id = std::stoi(next());
        else throw std::runtime_error("unknown argument: " + std::string(arg));
    }
    if (config.workdir.empty()) throw std::runtime_error("TT_RQM_PERSISTENT_QMUL_DIR is required");
    if (config.manifest.empty()) config.manifest = config.workdir / "manifest.json";
    if (config.device_id != 0) throw std::runtime_error("persistent Stage B candidate is restricted to Wormhole device 0");
    return config;
}

json run_case(PersistentDeviceSession& session, const std::filesystem::path& workdir, const json& spec) {
    const uint32_t items = spec.at("items").get<uint32_t>();
    const uint32_t iterations = spec.at("iterations").get<uint32_t>();
    const uint32_t warmup = spec.at("warmup").get<uint32_t>();
    const uint32_t samples = spec.at("samples").get<uint32_t>();
    if (items == 0 || iterations == 0 || samples == 0) throw std::runtime_error("items, iterations, and samples must be positive");

    const auto& inputs = spec.at("inputs");
    const auto& outputs = spec.at("outputs");
    const auto a_aos = read_words(workdir / inputs.at("a").get<std::string>(), items);
    const auto b_aos = read_words(workdir / inputs.at("b").get<std::string>(), items);
    auto& cq = session.command_queue();
    const uint32_t component_tiles = (items + kElementsPerTile - 1) / kElementsPerTile;
    const uint32_t total_bytes = kLanes * component_tiles * kTileBytes;

    const auto allocation_started = Clock::now();
    distributed::DeviceLocalBufferConfig local_config{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_bytes};
    auto a = distributed::MeshBuffer::create(buffer_config, local_config, session.device().get());
    auto b = distributed::MeshBuffer::create(buffer_config, local_config, session.device().get());
    auto out = distributed::MeshBuffer::create(buffer_config, local_config, session.device().get());
    const double buffer_allocation_s = seconds_since(allocation_started);

    const auto build_started = Clock::now();
    auto prepared = build_workload(session.device(), a, b, out, component_tiles);
    const double program_build_s = seconds_since(build_started);

    const auto a_planar = aos_to_planar_tiles(a_aos, items);
    const auto b_planar = aos_to_planar_tiles(b_aos, items);
    const auto h2d_started = Clock::now();
    distributed::EnqueueWriteMeshBuffer(cq, a, a_planar, false);
    distributed::EnqueueWriteMeshBuffer(cq, b, b_planar, false);
    distributed::Finish(cq);
    const double h2d_s = seconds_since(h2d_started);

    const auto prewarm_started = Clock::now();
    distributed::Finish(cq);
    const double prewarm_sync_s = seconds_since(prewarm_started);

    const auto warmup_started = Clock::now();
    for (uint32_t i = 0; i < warmup; ++i) distributed::EnqueueMeshWorkload(cq, prepared.workload, true);
    distributed::Finish(cq);
    const double warmup_s = seconds_since(warmup_started);

    std::vector<double> sample_timings;
    sample_timings.reserve(samples);
    for (uint32_t sample = 0; sample < samples; ++sample) {
        const auto sample_started = Clock::now();
        for (uint32_t i = 0; i < iterations; ++i) distributed::EnqueueMeshWorkload(cq, prepared.workload, true);
        distributed::Finish(cq);
        sample_timings.push_back(seconds_since(sample_started));
    }

    const auto d2h_started = Clock::now();
    std::vector<uint32_t> out_planar;
    distributed::EnqueueReadMeshBuffer(cq, out_planar, out, true);
    distributed::Finish(cq);
    const double d2h_s = seconds_since(d2h_started);
    const auto out_aos = planar_tiles_to_aos(out_planar, items);
    write_words(workdir / outputs.at("out").get<std::string>(), out_aos);

    const auto cleanup_started = Clock::now();
    out.reset();
    b.reset();
    a.reset();
    const double cleanup_s = seconds_since(cleanup_started);

    return {
        {"case_id", spec.at("case_id")},
        {"items", items}, {"iterations", iterations}, {"warmup", warmup}, {"samples", samples},
        {"input_identity", {{"a_sha256", inputs.at("a_sha256")}, {"b_sha256", inputs.at("b_sha256")}}},
        {"output_identity", {{"fnv1a64", fnv1a64(out_aos)}, {"value_count", out_aos.size()}}},
        {"timings_s", {
            {"buffer_allocation", buffer_allocation_s}, {"program_build", program_build_s},
            {"h2d", h2d_s}, {"prewarm_sync", prewarm_sync_s}, {"warmup", warmup_s},
            {"samples", sample_timings}, {"d2h", d2h_s}, {"cleanup", cleanup_s}
        }},
        {"work", {
            {"device_count", 1}, {"device_id", 0}, {"core_count", prepared.metadata.core_count},
            {"component_tiles", prepared.metadata.component_tiles}, {"grid_x", prepared.metadata.grid_x},
            {"grid_y", prepared.metadata.grid_y}, {"available_core_count", prepared.metadata.grid_x * prepared.metadata.grid_y},
            {"layout", "planar_float32_tiles_32x32"}, {"work_split", "row_major"},
            {"arithmetic_path", "tensix_compute_sfpu"}
        }}
    };
}

int run(int argc, char** argv) {
    const auto process_started = Clock::now();
    const auto config = parse_persistent_config(argc, argv);
    const json manifest = json::parse(read_text(config.manifest));
    if (manifest.value("schema", "") != kPersistentProtocol || manifest.value("workload", "") != "qmul" ||
        manifest.value("dtype", "") != "float32") throw std::runtime_error("unsupported persistent-qmul manifest");

    json case_metrics = json::array();
    double create_s = 0.0;
    double close_s = 0.0;
    uint32_t create_count = 0;
    uint32_t close_count = 0;
    {
        PersistentDeviceSession session(config.device_id);
        create_s = session.create_s();
        create_count = session.create_count();
        for (const auto& spec : manifest.at("cases")) case_metrics.push_back(run_case(session, config.workdir, spec));
        close_s = session.close();
        close_count = session.close_count();
    }

    json metrics = {
        {"schema", std::string(kPersistentMetrics)}, {"protocol", std::string(kPersistentProtocol)},
        {"backend", "tt-metalium-qmul-multicore-persistent-candidate"},
        {"device", "tenstorrent/wormhole-device-0"}, {"dtype", "float32"},
        {"execution_kind", execution_kind()}, {"implementation_class", std::string(kImplementationClass)},
        {"performance_eligible", true}, {"stable_benchmark", false},
        {"lifecycle", {{"device_count", 1}, {"device_id", 0}, {"create_count", create_count}, {"close_count", close_count}}},
        {"session_timings_s", {{"device_create", create_s}, {"device_close", close_s}, {"candidate_session", seconds_since(process_started)}}},
        {"cases", case_metrics},
        {"provenance", {
            {"chip_type", env_or_unknown("TT_RQM_CHIP_TYPE")},
            {"tt_metal_commit", env_or_unknown("TT_RQM_TT_METAL_COMMIT")},
            {"compiler_version", env_or_unknown("TT_RQM_COMPILER_VERSION")},
            {"runtime_version", env_or_unknown("TT_RQM_RUNTIME_VERSION")},
            {"build_id", env_or_unknown("TT_RQM_BUILD_ID")},
            {"candidate_sha256", env_or_unknown("TT_RQM_CANDIDATE_SHA256")},
            {"repository_commit", env_or_unknown("TT_RQM_REPOSITORY_COMMIT")},
            {"timer_scope", "one persistent device session; each sample is prepared workload enqueue plus Finish"}
        }}
    };
    write_text(config.workdir / manifest.at("outputs").at("metrics").get<std::string>(), metrics.dump(2) + "\n");
    return 0;
}

}  // namespace persistent_qmul

int main(int argc, char** argv) {
    try {
        return persistent_qmul::run(argc, argv);
    } catch (const std::exception& exc) {
        std::cerr << "tt_rqm_metalium_qmul_multicore_persistent_candidate failed: " << exc.what() << "\n";
        return 2;
    }
}
