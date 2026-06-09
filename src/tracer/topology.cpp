#include "tracer/topology.hpp"

#include "llama.h"

namespace trace {

int Topology::add_node(TopologyNode node) {
    const int idx = static_cast<int>(nodes_.size());
    nodes_.push_back(std::move(node));
    return idx;
}

void Topology::build_from_model(const llama_model* model, const std::string& model_name) {
    nodes_.clear();
    capture_target_ = 0;

    TopologyNode root;
    root.id = "model";
    root.label = model_name.empty() ? "local-model" : model_name;
    root.depth = 0;
    root.expandable = true;
    root.expanded = true;
    const int root_idx = add_node(std::move(root));

    TopologyNode embed;
    embed.id = "embed_tokens";
    embed.label = "embed_tokens";
    embed.depth = 1;
    nodes_[root_idx].children.push_back(add_node(std::move(embed)));

    TopologyNode layers_group;
    layers_group.id = "layers";
    layers_group.label = "layers";
    layers_group.depth = 1;
    layers_group.expandable = true;
    layers_group.expanded = true;
    const int layers_idx = add_node(std::move(layers_group));
    nodes_[root_idx].children.push_back(layers_idx);

    const int n_layer = model ? llama_n_layer(model) : 4;
    for (int i = 0; i < n_layer; ++i) {
        TopologyNode layer;
        layer.id = "layers." + std::to_string(i);
        layer.label = "layers." + std::to_string(i);
        layer.depth = 2;
        layer.layer_index = i;
        layer.expandable = true;
        layer.expanded = (i <= 1);
        const int layer_idx = add_node(std::move(layer));

        TopologyNode attn;
        attn.id = "layers." + std::to_string(i) + ".attn";
        attn.label = "layers." + std::to_string(i) + ".attn";
        attn.depth = 3;
        attn.layer_index = i;
        const int attn_idx = add_node(std::move(attn));

        TopologyNode mlp;
        mlp.id = "layers." + std::to_string(i) + ".mlp";
        mlp.label = "layers." + std::to_string(i) + ".mlp";
        mlp.depth = 3;
        mlp.layer_index = i;
        const int mlp_idx = add_node(std::move(mlp));

        nodes_[layer_idx].children = {attn_idx, mlp_idx};
        nodes_[layers_idx].children.push_back(layer_idx);
    }

  if (!nodes_.empty()) {
        set_capture_target(0);
    }
}

void Topology::set_capture_target(int flat_index) {
    if (flat_index < 0 || flat_index >= static_cast<int>(nodes_.size())) {
        return;
    }
    for (auto& n : nodes_) {
        n.is_capture_target = false;
    }
    nodes_[flat_index].is_capture_target = true;
    capture_target_ = flat_index;
}

}  // namespace trace
