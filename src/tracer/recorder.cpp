#include "tracer/recorder.hpp"

#include <cstdint>
#include <fstream>

namespace trace {

namespace {

enum class RecordType : uint8_t {
    Event = 1,
    Anomaly = 2,
    Attention = 3,
};

constexpr int64_t steady_to_ns(const std::chrono::steady_clock::time_point& tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

constexpr std::chrono::steady_clock::time_point ns_to_steady(int64_t ns) {
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}

}  // namespace

Recorder::Recorder(std::string path) : path_(std::move(path)) {}

bool Recorder::open() {
    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (!out_) {
        return false;
    }
    out_.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    out_.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    return true;
}

void Recorder::close() {
    if (out_.is_open()) {
        out_.close();
    }
}

void Recorder::write_string(const std::string& s) {
    const uint32_t len = static_cast<uint32_t>(s.size());
    out_.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out_.write(s.data(), static_cast<std::streamsize>(s.size()));
}

void Recorder::write_event(const TraceEvent& ev) {
    if (!out_) {
        return;
    }
    const auto type = static_cast<uint8_t>(RecordType::Event);
    out_.write(reinterpret_cast<const char*>(&type), sizeof(type));

    out_.write(reinterpret_cast<const char*>(&ev.id), sizeof(ev.id));
    const int64_t ts = steady_to_ns(ev.timestamp);
    out_.write(reinterpret_cast<const char*>(&ts), sizeof(ts));

    write_string(ev.tensor_name);
    write_string(ev.op_name);
    const auto layer_type = static_cast<uint8_t>(ev.layer_type);
    out_.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));
    out_.write(reinterpret_cast<const char*>(&ev.layer_index), sizeof(ev.layer_index));
    out_.write(reinterpret_cast<const char*>(ev.shape.data()),
               static_cast<std::streamsize>(ev.shape.size() * sizeof(int64_t)));
    out_.write(reinterpret_cast<const char*>(&ev.ggml_type), sizeof(ev.ggml_type));
    write_string(ev.device);
    out_.write(reinterpret_cast<const char*>(&ev.stats), sizeof(ev.stats));
    out_.write(reinterpret_cast<const char*>(&ev.latency_ms), sizeof(ev.latency_ms));
    out_.write(reinterpret_cast<const char*>(&ev.block_latency_ms), sizeof(ev.block_latency_ms));
}

void Recorder::write_anomaly(const AnomalyRecord& rec) {
    if (!out_) {
        return;
    }
    const auto type = static_cast<uint8_t>(RecordType::Anomaly);
    out_.write(reinterpret_cast<const char*>(&type), sizeof(type));
    const int64_t ts = steady_to_ns(rec.timestamp);
    out_.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
    const auto kind = static_cast<uint8_t>(rec.kind);
    out_.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
    write_string(rec.message);
    out_.write(reinterpret_cast<const char*>(&rec.layer_index), sizeof(rec.layer_index));
}

void Recorder::write_attention(const AttentionSnapshot& snap) {
    if (!out_) {
        return;
    }
    const auto type = static_cast<uint8_t>(RecordType::Attention);
    out_.write(reinterpret_cast<const char*>(&type), sizeof(type));
    out_.write(reinterpret_cast<const char*>(&snap.layer), sizeof(snap.layer));
    out_.write(reinterpret_cast<const char*>(&snap.n_heads), sizeof(snap.n_heads));
    out_.write(reinterpret_cast<const char*>(&snap.n_tokens), sizeof(snap.n_tokens));
    const int64_t ts = steady_to_ns(snap.timestamp);
    out_.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
    const uint32_t count = static_cast<uint32_t>(snap.weights.size());
    out_.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out_.write(reinterpret_cast<const char*>(snap.weights.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
}

Replayer::Replayer(std::string path) : path_(std::move(path)) {}

std::string Replayer::read_string(std::ifstream& in) {
    uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(len, '\0');
    in.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}

bool Replayer::load() {
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != Recorder::kMagic || version != Recorder::kVersion) {
        return false;
    }

    events_.clear();
    anomalies_.clear();
    attentions_.clear();

    while (in.peek() != EOF) {
        uint8_t type = 0;
        in.read(reinterpret_cast<char*>(&type), sizeof(type));
        if (!in) {
            break;
        }

        if (static_cast<RecordType>(type) == RecordType::Event) {
            TraceEvent ev;
            in.read(reinterpret_cast<char*>(&ev.id), sizeof(ev.id));
            int64_t ts = 0;
            in.read(reinterpret_cast<char*>(&ts), sizeof(ts));
            ev.timestamp = ns_to_steady(ts);
            ev.tensor_name = read_string(in);
            ev.op_name = read_string(in);
            uint8_t layer_type = 0;
            in.read(reinterpret_cast<char*>(&layer_type), sizeof(layer_type));
            ev.layer_type = static_cast<LayerType>(layer_type);
            in.read(reinterpret_cast<char*>(&ev.layer_index), sizeof(ev.layer_index));
            in.read(reinterpret_cast<char*>(ev.shape.data()),
                    static_cast<std::streamsize>(ev.shape.size() * sizeof(int64_t)));
            in.read(reinterpret_cast<char*>(&ev.ggml_type), sizeof(ev.ggml_type));
            ev.device = read_string(in);
            in.read(reinterpret_cast<char*>(&ev.stats), sizeof(ev.stats));
            in.read(reinterpret_cast<char*>(&ev.latency_ms), sizeof(ev.latency_ms));
            in.read(reinterpret_cast<char*>(&ev.block_latency_ms), sizeof(ev.block_latency_ms));
            events_.push_back(std::move(ev));
        } else if (static_cast<RecordType>(type) == RecordType::Anomaly) {
            AnomalyRecord rec;
            int64_t ts = 0;
            in.read(reinterpret_cast<char*>(&ts), sizeof(ts));
            rec.timestamp = ns_to_steady(ts);
            uint8_t kind = 0;
            in.read(reinterpret_cast<char*>(&kind), sizeof(kind));
            rec.kind = static_cast<AnomalyKind>(kind);
            rec.message = read_string(in);
            in.read(reinterpret_cast<char*>(&rec.layer_index), sizeof(rec.layer_index));
            anomalies_.push_back(std::move(rec));
        } else if (static_cast<RecordType>(type) == RecordType::Attention) {
            AttentionSnapshot snap;
            in.read(reinterpret_cast<char*>(&snap.layer), sizeof(snap.layer));
            if (version >= 2) {
                in.read(reinterpret_cast<char*>(&snap.n_heads), sizeof(snap.n_heads));
            } else {
                int legacy_head = 0;
                in.read(reinterpret_cast<char*>(&legacy_head), sizeof(legacy_head));
                snap.n_heads = std::max(1, legacy_head + 1);
            }
            in.read(reinterpret_cast<char*>(&snap.n_tokens), sizeof(snap.n_tokens));
            int64_t ts = 0;
            in.read(reinterpret_cast<char*>(&ts), sizeof(ts));
            snap.timestamp = ns_to_steady(ts);
            uint32_t count = 0;
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            snap.weights.resize(count);
            in.read(reinterpret_cast<char*>(snap.weights.data()),
                    static_cast<std::streamsize>(count * sizeof(float)));
            if (version < 2 && snap.n_tokens > 0 && !snap.weights.empty()) {
                snap.n_heads = static_cast<int>(snap.weights.size()) /
                               (snap.n_tokens * snap.n_tokens);
                if (snap.n_heads < 1) {
                    snap.n_heads = 1;
                }
            }
            attentions_.push_back(std::move(snap));
        }
    }

    return !events_.empty();
}

void Replayer::push_into(EventRingBuffer& events, AnomalyRingBuffer& anomalies) const {
    for (const auto& ev : events_) {
        events.push(ev);
    }
    for (const auto& rec : anomalies_) {
        anomalies.push(rec);
    }
}

}  // namespace trace
