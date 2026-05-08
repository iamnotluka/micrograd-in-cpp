#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct MnistSample {
      std::vector<double> pixels;
      int label;
};

int read_big_endian_int(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);

    if (!file) {
        throw std::runtime_error("Failed to read integer from file");
    }

    return (bytes[0] << 24)
        |  (bytes[1] << 16)
        |  (bytes[2] << 8)
        |  bytes[3];
}

std::vector<std::vector<double>> load_images(const std::string& path, int max_images = -1) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(std::string("Could not open image file: ") + path);
    }

    int magic_number = read_big_endian_int(file);
    int image_count = read_big_endian_int(file);
    int rows = read_big_endian_int(file);
    int cols = read_big_endian_int(file);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file");
    }

    int images_to_read = image_count;
    if (max_images > 0 && max_images < image_count) {
        images_to_read = max_images;
    }

    std::vector<std::vector<double>> images;
    images.reserve(images_to_read);

    int pixels_per_image = rows * cols;

    for (int i = 0; i < images_to_read; i++) {
        std::vector<double> pixels;
        pixels.reserve(pixels_per_image);

        for (int j = 0; j < pixels_per_image; j++) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);

            double normalised_pixel = static_cast<double>(pixel) / 255.0;
            pixels.push_back(normalised_pixel);
        }

        images.push_back(pixels);
    }

    return images;
}

std::vector<int> load_labels(const std::string& path, int max_labels = -1) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(std::string("Could not open label file: ") + path);
    }

    int magic_number = read_big_endian_int(file);
    int label_count = read_big_endian_int(file);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file");
    }

    int labels_to_read = label_count;
    if (max_labels > 0 && max_labels < label_count) {
        labels_to_read = max_labels;
    }

    std::vector<int> labels;
    labels.reserve(labels_to_read);

    for (int i = 0; i< labels_to_read; i++) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }

    return labels;
}

std::vector<MnistSample> load_mnist_dataset(
    const std::string& image_path,
    const std::string& label_path,
    int max_samples = -1
) {
    auto images = load_images(image_path, max_samples);
    auto labels = load_labels(label_path, max_samples);

    if (images.size() != labels.size()) {
        throw std::runtime_error("Image count does not match label count.");
    } 

    std::vector<MnistSample> samples;
    samples.reserve(images.size());

    for (size_t i = 0; i < images.size(); i++) {
        MnistSample sample;
        sample.pixels = images[i];
        sample.label = labels[i];

        samples.push_back(sample);
    }

    return samples;
}
