#include "inference_mlp.h"
#include "mnist_display.h"
#include "mnist_loader.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

void print_scores(const std::vector<double>& scores) {
    std::cout << "scores:";

    for (size_t i = 0; i < scores.size(); i++) {
        std::cout << " " << i << "=" << std::fixed << std::setprecision(2) << scores[i];
    }

    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    try {
        int sample_limit = 1000;
        int print_limit = 20;

        if (argc > 1) {
            sample_limit = std::stoi(argv[1]);
        }

        if (argc > 2) {
            print_limit = std::stoi(argv[2]);
        }

        if (sample_limit < 1) {
            throw std::runtime_error("Sample limit must be at least 1");
        }

        auto model = InferenceMLP::load("models/digits_mlp.params");

        auto samples = load_mnist_dataset(
            "data/mnist/t10k-images-idx3-ubyte",
            "data/mnist/t10k-labels-idx1-ubyte",
            sample_limit
        );

        int correct_count = 0;
        print_limit = std::min(print_limit, static_cast<int>(samples.size()));

        std::cout << "Loaded model from models/digits_mlp.params\n";
        std::cout << "Loaded " << samples.size() << " MNIST test samples\n\n";

        for (size_t i = 0; i < samples.size(); i++) {
            const auto& sample = samples[i];
            auto result = model.forward(sample.pixels);
            int prediction = InferenceMLP::argmax(result.outputs);

            if (prediction == sample.label) {
                correct_count++;
            }

            if (static_cast<int>(i) < print_limit) {
                std::cout << "sample " << i + 1
                          << " | actual: " << sample.label
                          << " | predicted: " << prediction;

                if (prediction == sample.label) {
                    std::cout << " | correct\n";
                } else {
                    std::cout << " | wrong\n";
                }

                print_digit_image(sample);
                print_scores(result.outputs);
                std::cout << "\n";
            }
        }

        double accuracy = static_cast<double>(correct_count) / samples.size();
        std::cout << "Accuracy: " << accuracy * 100.0 << "% "
                  << "(" << correct_count << "/" << samples.size() << ")\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
