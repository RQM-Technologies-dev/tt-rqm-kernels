// Development-only H2B D0-D4 runtime-isolation executable. D5 is executed by
// the Python runner through the existing external H2B protocol binary.
#define main h2b_candidate_main
#include H2B_CANDIDATE_SOURCE
#undef main

#include <algorithm>
#include <csignal>
#include <optional>
#include <unistd.h>

namespace {

constexpr std::string_view kEventSchema = "tt-rqm-h2b-runtime-isolation-event.v1";
constexpr std::string_view kPrefix = "H2B_RUNTIME_EVENT ";
const std::vector<std::string> kStages = {"d0", "d1", "d2", "d3", "d4"};

struct Events {
    std::string diagnostic_id;
    std::string stage;
    Clock::time_point start = Clock::now();

    void emit(
        std::string_view event,
        std::string_view status,
        int device_id = -1,
        std::string_view error = {},
        json validation = nullptr) const {
        json record = {
            {"schema", kEventSchema}, {"diagnostic_id", diagnostic_id},
            {"stage", stage}, {"event", event}, {"status", status},
            {"elapsed_monotonic_s", elapsed(start)}, {"process_id", ::getpid()},
            {"device_id", device_id < 0 ? json(nullptr) : json(device_id)},
            {"error", error.empty() ? json(nullptr) : json(error)},
        };
        if (!validation.is_null()) record["validation"] = std::move(validation);
        std::cerr << kPrefix << record.dump() << '\n' << std::flush;
    }
    void begin(std::string_view event, int device_id = -1) const {
        emit(event, "started", device_id);
    }
    void end(std::string_view event, int device_id = -1, json validation = nullptr) const {
        emit(event, "completed", device_id, {}, std::move(validation));
    }
};

std::shared_ptr<distributed::MeshBuffer> diagnostic_buffer(
    const std::shared_ptr<distributed::MeshDevice>& device, uint32_t bytes) {
    distributed::DeviceLocalBufferConfig local{
        .page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = bytes}, local, device.get());
}

void validate_identity_planes(const std::vector<uint32_t>& words, uint32_t steps) {
    const float w = std::bit_cast<float>(words[0]);
    const float x = std::bit_cast<float>(words[kTileElements]);
    const float y = std::bit_cast<float>(words[2 * kTileElements]);
    const float z = std::bit_cast<float>(words[3 * kTileElements]);
    const float phase_real = std::bit_cast<float>(words[4 * kTileElements]);
    const float phase_imag = std::bit_cast<float>(words[5 * kTileElements]);
    (void)steps;
    if (std::abs(w - 1.0f) > 1e-5f || std::abs(x) > 1e-5f ||
        std::abs(y) > 1e-5f || std::abs(z) > 1e-5f ||
        std::abs(phase_real - 1.0f) > 1e-5f || std::abs(phase_imag) > 1e-5f) {
        throw std::runtime_error("identity oracle validation failed");
    }
}

void close_device(const Events& events, const std::shared_ptr<distributed::MeshDevice>& device) {
    events.begin("device_close_begin", 0);
    if (!device->close()) throw std::runtime_error("failed to close MeshDevice");
    events.end("device_close_end", 0);
}

int run_d0(const Events& events) {
    events.begin("device_create_begin", 0);
    auto device = distributed::MeshDevice::create_unit_mesh(0);
    events.end("device_create_end", 0);
    events.begin("queue_acquire_begin", 0);
    (void)device->mesh_command_queue();
    events.end("queue_acquire_end", 0);
    close_device(events, device);
    return 0;
}

int run_d1(const Events& events) {
    events.begin("device_create_begin", 0);
    auto device = distributed::MeshDevice::create_unit_mesh(0);
    events.end("device_create_end", 0);
    events.begin("queue_acquire_begin", 0);
    auto& queue = device->mesh_command_queue();
    events.end("queue_acquire_end", 0);
    events.begin("buffer_allocate_begin", 0);
    auto buffer = diagnostic_buffer(device, kTileBytes);
    events.end("buffer_allocate_end", 0);
    std::vector<uint32_t> input(kTileElements);
    for (uint32_t index = 0; index < kTileElements; ++index) input[index] = 0xA5000000U ^ index;
    events.begin("h2d_begin", 0);
    distributed::EnqueueWriteMeshBuffer(queue, buffer, input, false);
    distributed::Finish(queue);
    events.end("h2d_end", 0);
    events.begin("d2h_begin", 0);
    std::vector<uint32_t> output;
    distributed::EnqueueReadMeshBuffer(queue, output, buffer, true);
    distributed::Finish(queue);
    events.end("d2h_end", 0);
    events.begin("validation_begin", 0);
    if (output != input) throw std::runtime_error("D1 DRAM loopback bitwise mismatch");
    events.end("validation_end", 0, {{"passed", true}, {"kind", "bitwise_dram_loopback"}});
    close_device(events, device);
    return 0;
}

int run_kernel_stage(const Events& events, std::string_view stage) {
    constexpr uint32_t steps = 1;
    constexpr uint32_t component_tiles = 1;
    constexpr uint32_t bytes = steps * kLanes * component_tiles * kTileBytes;
    events.begin("device_create_begin", 0);
    auto device = distributed::MeshDevice::create_unit_mesh(0);
    events.end("device_create_end", 0);
    events.begin("queue_acquire_begin", 0);
    auto& queue = device->mesh_command_queue();
    events.end("queue_acquire_end", 0);
    events.begin("buffer_allocate_begin", 0);
    auto input = diagnostic_buffer(device, bytes);
    auto intermediate = diagnostic_buffer(device, bytes);
    auto final_output = diagnostic_buffer(device, bytes);
    events.end("buffer_allocate_end", 0);

    const bool use_h2a = stage != "d3";
    const bool use_h1 = stage != "d2";
    std::optional<PreparedProgram> h2a;
    std::optional<PreparedProgram> h1;
    if (use_h2a) {
        events.begin("program_build_h2a_begin", 0);
        h2a.emplace(build_h2a_program(device, input, intermediate, component_tiles, steps));
        events.end("program_build_h2a_end", 0);
    }
    if (use_h1) {
        events.begin("program_build_h1_begin", 0);
        h1.emplace(build_h1_program(device, intermediate, final_output, component_tiles, steps));
        events.end("program_build_h1_end", 0);
    }

    std::vector<uint32_t> host_input(bytes / sizeof(uint32_t), 0);
    auto destination = input;
    if (stage == "d3") {
        destination = intermediate;
        host_input[0] = std::bit_cast<uint32_t>(1.0f);
        host_input[4 * kTileElements] = std::bit_cast<uint32_t>(1.0f);
    } else {
        host_input[5 * kTileElements] = std::bit_cast<uint32_t>(1.0f);
    }
    events.begin("h2d_begin", 0);
    distributed::EnqueueWriteMeshBuffer(queue, destination, host_input, false);
    distributed::Finish(queue);
    events.end("h2d_end", 0);
    if (use_h2a) {
        events.begin("execute_h2a_begin", 0);
        distributed::EnqueueMeshWorkload(queue, h2a->workload, false);
        distributed::Finish(queue);
        events.end("execute_h2a_end", 0);
    }
    if (use_h1) {
        events.begin("execute_h1_begin", 0);
        distributed::EnqueueMeshWorkload(queue, h1->workload, false);
        distributed::Finish(queue);
        events.end("execute_h1_end", 0);
    }
    events.begin("d2h_begin", 0);
    std::vector<uint32_t> output;
    auto source = use_h1 ? final_output : intermediate;
    distributed::EnqueueReadMeshBuffer(queue, output, source, true);
    distributed::Finish(queue);
    events.end("d2h_end", 0);
    events.begin("validation_begin", 0);
    validate_identity_planes(output, steps);
    events.end("validation_end", 0, {{"passed", true}, {"kind", "identity_oracle"}});
    close_device(events, device);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string stage;
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string_view(argv[index]) == "--stage") stage = argv[index + 1];
    }
    const char* id = std::getenv("TT_RQM_H2B_DIAGNOSTIC_ID");
    Events events{id == nullptr ? "missing-diagnostic-id" : id, stage};
    events.end("process_start");
    try {
        if (std::find(kStages.begin(), kStages.end(), stage) == kStages.end())
            throw std::runtime_error("unknown H2B runtime-isolation stage");
        int result = 0;
        if (stage == "d0") result = run_d0(events);
        else if (stage == "d1") result = run_d1(events);
        else result = run_kernel_stage(events, stage);
        events.end("process_end");
        return result;
    } catch (const std::exception& error) {
        events.emit("failure", "failed", 0, error.what());
        return 2;
    }
}
