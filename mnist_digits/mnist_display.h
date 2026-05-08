#pragma once

#include "mnist_loader.h"

#include <iostream>

void print_digit_image(const MnistSample& sample) {
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            double pixel = sample.pixels[row * 28 + col];

            if (pixel > 0.75) {
                std::cout << "#";
            } else if (pixel > 0.35) {
                std::cout << "+";
            } else if (pixel > 0.10) {
                std::cout << ".";
            } else {
                std::cout << " ";
            }
        }

        std::cout << "\n";
    }
}

void print_digit_with_label(const MnistSample& sample) {
    std::cout << "Label: " << sample.label << "\n\n";
    print_digit_image(sample);
}
