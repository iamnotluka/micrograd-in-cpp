#include "mnist_display.h"
#include "mnist_loader.h"

#include <string>

void clear_screen() {
    std::cout << "\033[2J\033[H";
}

void inspect_samples(const std::vector<MnistSample>& samples) {
    while (true) {
        std::cout << "Enter sample number 1-" << samples.size() << " (q to quit): ";

        std::string input;
        if (!(std::cin >> input)) {
            break;
        }

        if (input == "q") {
            break;
        }

        int sample_number = 0;
        try {
            sample_number = std::stoi(input);
        } catch (const std::exception&) {
            clear_screen();
            std::cout << "Please enter a number or q.\n\n";
            continue;
        }

        if (sample_number < 1 || sample_number > static_cast<int>(samples.size())) {
            clear_screen();
            std::cout << "Sample number must be between 1 and " << samples.size() << ".\n\n";
            continue;
        }

        size_t sample_index = static_cast<size_t>(sample_number - 1);

        clear_screen();
        std::cout << "Sample " << sample_number << " of " << samples.size() << "\n";
        print_digit_with_label(samples[sample_index]);
        std::cout << "\n";
    }
}

int main() {
    try {
        std::string image_path = "data/mnist/train-images-idx3-ubyte";
        std::string label_path = "data/mnist/train-labels-idx1-ubyte";

        auto samples = load_mnist_dataset(image_path, label_path);

        std::cout << "Loaded " << samples.size() << " MNIST samples\n\n";

        inspect_samples(samples);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
