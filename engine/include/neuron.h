#include <vector>
#include <memory>
#include <value.h>
#include <cmath>
#include <random>
#include <stdexcept>

class Neuron {
    public:
        Neuron(int number_of_inputs) {            
            std::random_device rd;
            std::mt19937 gen(rd());
            double limit = 1.0 / std::sqrt(number_of_inputs);
            std::uniform_real_distribution<double> dist(-limit, limit);

            for (int i = 0; i < number_of_inputs; i++) {
                weights_.push_back(std::make_shared<Value>(dist(gen)));
            }
            bias_ = std::make_shared<Value>(dist(gen));
        }

        std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> input) {
            int input_size = input.size();
            if (input_size != weights_.size()) {
                throw std::invalid_argument("Input size must match number of weights");
            };

            auto activation = Value::create(0.0);

            for (int i = 0; i < input_size; i++) {
                activation = activation + weights_[i]*input[i];
            };

            activation = activation + bias_;

            return activation->tanh(activation);
        }

        std::vector<std::shared_ptr<Value>> parameters() {
            std::vector<std::shared_ptr<Value>> parameters = weights_;
            parameters.push_back(bias_);
            return parameters;
        }

    private:
        std::vector<std::shared_ptr<Value>> weights_;
        std::shared_ptr<Value> bias_;
};
