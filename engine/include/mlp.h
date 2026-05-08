#include <layer.h>
#include <vector>

class MLP {
    public:
        MLP(int number_of_inputs, std::vector<int> layer_outputs) {
            std::vector<int> sizes = {number_of_inputs};
            sizes.insert(sizes.end(), layer_outputs.begin(), layer_outputs.end());
            auto number_of_layers = layer_outputs.size();

            for (int i = 0; i < number_of_layers; i++) {
                layers_.push_back(std::make_shared<Layer>(sizes[i], sizes[i+1]));
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> input_values) {
            for (int i = 0; i < layers_.size(); i++) {
                input_values = (*layers_[i])(input_values);
            }

            return input_values;
        }

    private:
        std::vector<std::shared_ptr<Layer>> layers_;
};