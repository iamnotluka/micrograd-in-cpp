#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct InferenceNeuron {
    std::vector<double> weights;
    double bias;
};

struct InferenceLayer {
    std::vector<InferenceNeuron> neurons;
};

struct InferenceResult {
    std::vector<double> outputs;
    std::vector<std::vector<double>> layer_activations;
};

class InferenceMLP {
    public:
        static InferenceMLP load(const std::string& path) {
            std::ifstream file(path);

            if (!file) {
                throw std::runtime_error(std::string("Could not open model file: ") + path);
            }

            int architecture_size = 0;
            file >> architecture_size;

            if (architecture_size < 2) {
                throw std::runtime_error("Model architecture must have at least input and output sizes");
            }

            std::vector<int> architecture;
            architecture.reserve(architecture_size);

            for (int i = 0; i < architecture_size; i++) {
                int layer_size = 0;
                file >> layer_size;

                if (layer_size < 1) {
                    throw std::runtime_error("Model architecture contains an invalid layer size");
                }

                architecture.push_back(layer_size);
            }

            int parameter_count = 0;
            file >> parameter_count;

            int expected_parameter_count = count_parameters(architecture);
            if (parameter_count != expected_parameter_count) {
                throw std::runtime_error("Model parameter count does not match architecture");
            }

            InferenceMLP model(architecture);

            for (size_t layer_index = 0; layer_index < model.layers_.size(); layer_index++) {
                for (size_t neuron_index = 0; neuron_index < model.layers_[layer_index].neurons.size(); neuron_index++) {
                    auto& neuron = model.layers_[layer_index].neurons[neuron_index];

                    for (size_t weight_index = 0; weight_index < neuron.weights.size(); weight_index++) {
                        file >> neuron.weights[weight_index];
                    }

                    file >> neuron.bias;
                }
            }

            if (!file) {
                throw std::runtime_error("Failed while reading model parameters");
            }

            return model;
        }

        explicit InferenceMLP(const std::vector<int>& architecture)
            : architecture_(architecture) {
            if (architecture.size() < 2) {
                throw std::runtime_error("InferenceMLP needs at least input and output sizes");
            }

            for (size_t layer_index = 1; layer_index < architecture.size(); layer_index++) {
                int input_count = architecture[layer_index - 1];
                int output_count = architecture[layer_index];

                InferenceLayer layer;
                layer.neurons.reserve(output_count);

                for (int neuron_index = 0; neuron_index < output_count; neuron_index++) {
                    InferenceNeuron neuron;
                    neuron.weights.resize(input_count);
                    neuron.bias = 0.0;
                    layer.neurons.push_back(neuron);
                }

                layers_.push_back(layer);
            }
        }

        InferenceResult forward(const std::vector<double>& inputs) const {
            if (inputs.size() != static_cast<size_t>(architecture_[0])) {
                throw std::runtime_error("Input size does not match model architecture");
            }

            std::vector<double> current_values = inputs;
            std::vector<std::vector<double>> layer_activations;
            layer_activations.reserve(layers_.size());

            for (size_t layer_index = 0; layer_index < layers_.size(); layer_index++) {
                const auto& layer = layers_[layer_index];
                bool is_output_layer = layer_index == layers_.size() - 1;
                std::vector<double> next_values;
                next_values.reserve(layer.neurons.size());

                for (const auto& neuron : layer.neurons) {
                    double activation = neuron.bias;

                    for (size_t i = 0; i < neuron.weights.size(); i++) {
                        activation += neuron.weights[i] * current_values[i];
                    }

                    next_values.push_back(is_output_layer ? activation : std::tanh(activation));
                }

                layer_activations.push_back(next_values);
                current_values = next_values;
            }

            return {current_values, layer_activations};
        }

        int predict(const std::vector<double>& inputs) const {
            auto result = forward(inputs);
            return argmax(result.outputs);
        }

        const std::vector<int>& architecture() const {
            return architecture_;
        }

        static int argmax(const std::vector<double>& values) {
            if (values.empty()) {
                throw std::runtime_error("Cannot find argmax of an empty vector");
            }

            return static_cast<int>(
                std::max_element(values.begin(), values.end()) - values.begin()
            );
        }

    private:
        static int count_parameters(const std::vector<int>& architecture) {
            int total = 0;

            for (size_t layer_index = 1; layer_index < architecture.size(); layer_index++) {
                int input_count = architecture[layer_index - 1];
                int output_count = architecture[layer_index];
                total += output_count * (input_count + 1);
            }

            return total;
        }

        std::vector<int> architecture_;
        std::vector<InferenceLayer> layers_;
};
