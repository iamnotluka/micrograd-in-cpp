#include <layer.h>
#include <memory>
#include <stdexcept>
#include <vector>

class MLP {
    public:
        MLP(
            int number_of_inputs,
            std::vector<int> layer_outputs,
            Activation output_activation = Activation::Tanh
        ) {
            if (layer_outputs.empty()) {
                throw std::invalid_argument("MLP needs at least one output layer");
            }

            std::vector<int> sizes = {number_of_inputs};
            sizes.insert(sizes.end(), layer_outputs.begin(), layer_outputs.end());
            auto number_of_layers = layer_outputs.size();

            for (size_t i = 0; i < number_of_layers; i++) {
                Activation activation = i == number_of_layers - 1
                    ? output_activation
                    : Activation::Tanh;
                layers_.push_back(std::make_shared<Layer>(sizes[i], sizes[i+1], activation));
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> input_values) {
            for (int i = 0; i < layers_.size(); i++) {
                input_values = (*layers_[i])(input_values);
            }

            return input_values;
        }

        std::vector<std::shared_ptr<Value>> parameters() {
            std::vector<std::shared_ptr<Value>> parameters;
            for (int i = 0; i < layers_.size(); i++) {
                std::vector<std::shared_ptr<Value>> layer_parameters = layers_[i]->parameters();
                parameters.insert(parameters.end(), layer_parameters.begin(), layer_parameters.end());
            }
            return parameters;
        }
    private:
        std::vector<std::shared_ptr<Layer>> layers_;
};
