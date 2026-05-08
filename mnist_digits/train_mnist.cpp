#include "mnist_loader.h"

#include <mlp.h>
#include <value.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

std::vector<std::shared_ptr<Value>> pixels_to_values(const std::vector<double>& pixels) {
    std::vector<std::shared_ptr<Value>> values;
    values.reserve(pixels.size());

    for (double pixel : pixels) {
        values.push_back(Value::create(pixel));
    }

    return values;
}

std::vector<double> target_for_label(int label) {
    std::vector<double> target(10, -1.0);
    target[label] = 1.0;
    return target;
}

std::shared_ptr<Value> calculate_loss(
    const std::vector<std::shared_ptr<Value>>& predictions,
    const std::vector<double>& target
) {
    auto loss = Value::create(0.0);

    for (size_t i = 0; i < predictions.size(); i++) {
        auto diff = predictions[i] - target[i];
        loss = loss + diff->pow(diff, 2.0);
    }

    return loss / static_cast<double>(predictions.size());
}

int predicted_label(const std::vector<std::shared_ptr<Value>>& predictions) {
    int best_index = 0;
    double best_value = predictions[0]->data();

    for (size_t i = 1; i < predictions.size(); i++) {
        if (predictions[i]->data() > best_value) {
            best_value = predictions[i]->data();
            best_index = static_cast<int>(i);
        }
    }

    return best_index;
}

void zero_gradients(const std::vector<std::shared_ptr<Value>>& parameters) {
    for (const auto& parameter : parameters) {
        parameter->set_grad(0.0);
    }
}

void update_parameters(
    const std::vector<std::shared_ptr<Value>>& parameters,
    double learning_rate
) {
    for (const auto& parameter : parameters) {
        parameter->set_data(parameter->data() - learning_rate * parameter->grad());
    }
}

void save_parameters(
    const std::string& path,
    const std::vector<int>& architecture,
    const std::vector<std::shared_ptr<Value>>& parameters
) {
    std::ofstream file(path);

    if (!file) {
        throw std::runtime_error(std::string("Could not open model file: ") + path);
    }

    file << architecture.size() << "\n";

    for (int size : architecture) {
        file << size << " ";
    }

    file << "\n";
    file << parameters.size() << "\n";

    for (const auto& parameter : parameters) {
        file << parameter->data() << "\n";
    }
}

void train_model(
    MLP& model,
    const std::vector<MnistSample>& samples,
    int epochs,
    int training_limit,
    double learning_rate
) {
    training_limit = std::min(training_limit, static_cast<int>(samples.size()));
    if (training_limit < 1) {
        throw std::runtime_error("Training limit must be at least 1");
    }

    std::vector<int> sample_indices;
    sample_indices.reserve(training_limit);

    for (int i = 0; i < training_limit; i++) {
        sample_indices.push_back(i);
    }

    std::mt19937 rng(42);
    auto parameters = model.parameters();

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

        double total_loss = 0.0;
        int correct_count = 0;

        for (int sample_index : sample_indices) {
            const MnistSample& sample = samples[sample_index];

            auto input_values = pixels_to_values(sample.pixels);
            auto predictions = model(input_values);
            auto target = target_for_label(sample.label);
            auto loss = calculate_loss(predictions, target);

            total_loss += loss->data();

            if (predicted_label(predictions) == sample.label) {
                correct_count++;
            }

            zero_gradients(parameters);
            loss->backward();
            update_parameters(parameters, learning_rate);
        }

        double average_loss = total_loss / training_limit;
        double accuracy = static_cast<double>(correct_count) / training_limit;

        std::cout << "Epoch " << epoch + 1
                  << " | loss: " << average_loss
                  << " | accuracy: " << accuracy * 100.0 << "%\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        int training_limit = 50;
        int epochs = 2;
        double learning_rate = 0.01;

        if (argc > 1) {
            training_limit = std::stoi(argv[1]);
        }
        if (argc > 2) {
            epochs = std::stoi(argv[2]);
        }
        if (argc > 3) {
            learning_rate = std::stod(argv[3]);
        }

        auto samples = load_mnist_dataset(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
            training_limit
        );

        std::cout << "Loaded " << samples.size() << " MNIST samples\n" << std::flush;

        MLP model(784, {32, 10});
        auto parameters = model.parameters();

        std::cout << "Created MLP with 784 inputs, 32 hidden neurons, 10 outputs\n";
        std::cout << "Parameter count: " << parameters.size() << "\n";

        std::cout << "Training on " << training_limit
                  << " samples for " << epochs
                  << " epochs with learning rate " << learning_rate << "\n" << std::flush;

        train_model(model, samples, epochs, training_limit, learning_rate);

        save_parameters("models/digits_mlp.params", {784, 32, 10}, parameters);
        std::cout << "Saved model to models/digits_mlp.params\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
