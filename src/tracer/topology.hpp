#pragma once

#include <string>
#include <vector>

struct llama_model;

namespace trace {

struct TopologyNode {
    std::string id;
    std::string label;
    int depth = 0;
    int layer_index = -1;
    bool expandable = false;
    bool expanded = true;
    bool is_capture_target = false;
    std::vector<int> children;
};

class Topology {
public:
    void build_from_model(const llama_model* model, const std::string& model_name);
    const std::vector<TopologyNode>& nodes() const { return nodes_; }

    void set_capture_target(int flat_index);
    int capture_target() const { return capture_target_; }

private:
    int add_node(TopologyNode node);

    std::vector<TopologyNode> nodes_;
    int capture_target_ = 0;
};

}  // namespace trace
