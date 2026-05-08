#include <neuron.h>
#include <value.h>
#include <vector>
#include <memory>
#include <stdexcept>

class Layer {
    public:
        Layer(int number_of_inputs, int number_of_outputs) {
            for (int i = 0; i < number_of_outputs; i++) {
                neurons_.push_back(std::make_shared<Neuron>(number_of_inputs));
            }
            number_of_inputs_ = number_of_inputs;
        }

        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> input_values) {
            int input_size = input_values.size();
            if (input_size != number_of_inputs_) {
                throw std::invalid_argument("Input size must match the size of a neuron.");
            }

            std::vector<std::shared_ptr<Value>> outputs;
            for (int i = 0; i < neurons_.size(); i++) {
                outputs.push_back((*neurons_[i])(input_values));
            }

            return outputs;
        }

        std::vector<std::shared_ptr<Value>> parameters() {
            std::vector<std::shared_ptr<Value>> parameters;
            for (int i = 0; i < neurons_.size(); i++) {
                std::vector<std::shared_ptr<Value>> neuron_parameters = neurons_[i]->parameters();
                parameters.insert(parameters.end(), neuron_parameters.begin(), neuron_parameters.end());
            }
            return parameters;
        }

    private:
        std::vector<std::shared_ptr<Neuron>> neurons_;
        int number_of_inputs_;
};