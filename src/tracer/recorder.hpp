#pragma once

#include "tracer/event.hpp"
#include "tracer/ring_buffer.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace trace {

class Recorder {
public:
    explicit Recorder(std::string path);

    bool open();
    void close();
    bool is_open() const { return out_.is_open(); }

    void write_event(const TraceEvent& ev);
    void write_anomaly(const AnomalyRecord& rec);
    void write_attention(const AttentionSnapshot& snap);

    static constexpr uint32_t kMagic = 0x54524143;  // 'TRAC'
    static constexpr uint32_t kVersion = 2;

private:
    void write_string(const std::string& s);

    std::string path_;
    std::ofstream out_;
};

class Replayer {
public:
    explicit Replayer(std::string path);

    bool load();
    const std::vector<TraceEvent>& events() const { return events_; }
    const std::vector<AnomalyRecord>& anomalies() const { return anomalies_; }
    const std::vector<AttentionSnapshot>& attentions() const { return attentions_; }

    void push_into(trace::EventRingBuffer& events,
                   trace::AnomalyRingBuffer& anomalies) const;

private:
    std::string read_string(std::ifstream& in);

    std::string path_;
    std::vector<TraceEvent> events_;
    std::vector<AnomalyRecord> anomalies_;
    std::vector<AttentionSnapshot> attentions_;
};

}  // namespace trace
