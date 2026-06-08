#pragma once

#include "tracer/event.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

namespace trace {

// Single-producer single-consumer fixed-size ring buffer.
template <typename T, std::size_t Capacity>
class RingBuffer {
public:
    static_assert(Capacity > 0, "Capacity must be positive");

    bool push(T item) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t next = (tail + 1) % Capacity;
        if (next == head_.load(std::memory_order_acquire)) {
            // full: drop oldest
            head_.store((head_.load(std::memory_order_relaxed) + 1) % Capacity,
                        std::memory_order_release);
        }
        buffer_[tail] = std::move(item);
        tail_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        T item = std::move(buffer_[head]);
        head_.store((head + 1) % Capacity, std::memory_order_release);
        return item;
    }

    std::size_t size() const {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (tail >= head) {
            return tail - head;
        }
        return Capacity - head + tail;
    }

    bool empty() const { return size() == 0; }

    std::vector<T> snapshot() const {
        std::vector<T> out;
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (head == tail) {
            return out;
        }
        if (tail > head) {
            out.reserve(tail - head);
            for (std::size_t i = head; i < tail; ++i) {
                out.push_back(buffer_[i]);
            }
        } else {
            out.reserve(Capacity - head + tail);
            for (std::size_t i = head; i < Capacity; ++i) {
                out.push_back(buffer_[i]);
            }
            for (std::size_t i = 0; i < tail; ++i) {
                out.push_back(buffer_[i]);
            }
        }
        return out;
    }

    void clear() {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
    }

private:
    std::array<T, Capacity> buffer_{};
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};

constexpr std::size_t kDefaultEventCapacity = 512;
constexpr std::size_t kDefaultAnomalyCapacity = 128;

using EventRingBuffer = RingBuffer<TraceEvent, kDefaultEventCapacity>;
using AnomalyRingBuffer = RingBuffer<AnomalyRecord, kDefaultAnomalyCapacity>;

}  // namespace trace
