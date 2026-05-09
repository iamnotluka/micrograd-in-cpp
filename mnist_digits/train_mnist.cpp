#include "mnist_loader.h"

#include <loss.h>
#include <mlp.h>
#include <value.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <termios.h>
#include <unistd.h>

namespace {
    using Clock = std::chrono::steady_clock;
    const std::vector<int> DIGITS_ARCHITECTURE = {784, 32, 10};
    const std::string DEFAULT_MODEL_PATH = "models/digits_mlp.params";
    const std::string DEFAULT_CHECKPOINT_PATH = "models/digits_mlp.checkpoint";
    const std::string CHECKPOINT_MAGIC = "MICROGRAD_MNIST_CHECKPOINT";
    const int CHECKPOINT_VERSION = 2;

    struct PhaseTimers {
        double forward_seconds = 0.0;
        double loss_seconds = 0.0;
        double backward_seconds = 0.0;
        double update_seconds = 0.0;
    };

    struct EpochSummary {
        int epoch = 0;
        double loss = 0.0;
        double accuracy = 0.0;
        double seconds = 0.0;
    };

    struct EvaluationResult {
        int samples = 0;
        double loss = 0.0;
        double accuracy = 0.0;
        double seconds = 0.0;
    };

    struct TrainingState {
        int epoch = 1;
        int samples_completed_in_epoch = 0;
        long long samples_completed = 0;
        std::string phase = "starting";
        double epoch_loss = 0.0;
        int epoch_correct = 0;
        double last_loss = 0.0;
        int last_label = -1;
        int last_prediction = -1;
        std::string notice;
        bool resumed_from_checkpoint = false;
        double active_seconds_before_run = 0.0;
        PhaseTimers timers;
        std::vector<EpochSummary> completed_epochs;
        bool has_last_validation = false;
        int last_validation_epoch = 0;
        EvaluationResult last_validation;
    };

    struct CheckpointData {
        int epochs = 0;
        int training_limit = 0;
        double learning_rate = 0.0;
        int epoch_index = 0;
        int next_sample_position = 0;
        long long samples_completed = 0;
        double current_epoch_loss = 0.0;
        int current_epoch_correct = 0;
        double current_epoch_seconds = 0.0;
        double active_seconds = 0.0;
        double last_loss = 0.0;
        int last_label = -1;
        int last_prediction = -1;
        PhaseTimers timers;
        std::vector<int> architecture;
        std::vector<double> parameters;
        std::vector<int> sample_indices;
        std::vector<EpochSummary> completed_epochs;
        std::mt19937 rng;
    };

    struct TrainingOptions {
        int training_limit = 50;
        int epochs = 2;
        double learning_rate = 0.01;
        int validation_limit = 1000;
        int test_limit = 1000;
        bool fresh_start = false;
        std::string checkpoint_path = DEFAULT_CHECKPOINT_PATH;
    };

    struct TrainingResult {
        bool completed = false;
        bool checkpointed = false;
        bool stopped = false;
        std::string checkpoint_path;
    };

    class TerminalControls;

    TerminalControls* active_terminal_controls = nullptr;
    volatile std::sig_atomic_t requested_stop_signal = 0;
    void request_training_stop(int signal);

    class TerminalControls {
        public:
            TerminalControls() {
                active_terminal_controls = this;
                std::signal(SIGINT, request_training_stop);
                std::signal(SIGTERM, request_training_stop);

                if (!isatty(STDIN_FILENO)) {
                    return;
                }

                if (tcgetattr(STDIN_FILENO, &original_terminal_) != 0) {
                    return;
                }

                termios interactive_terminal = original_terminal_;
                interactive_terminal.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
                interactive_terminal.c_cc[VMIN] = 0;
                interactive_terminal.c_cc[VTIME] = 0;

                if (tcsetattr(STDIN_FILENO, TCSANOW, &interactive_terminal) != 0) {
                    return;
                }

                enabled_ = true;
            }

            ~TerminalControls() {
                restore();
                std::signal(SIGINT, SIG_DFL);
                std::signal(SIGTERM, SIG_DFL);

                if (active_terminal_controls == this) {
                    active_terminal_controls = nullptr;
                }
            }

            bool enabled() const {
                return enabled_;
            }

            char read_key() const {
                if (!enabled_) {
                    return '\0';
                }

                char key = '\0';
                ssize_t bytes_read = read(STDIN_FILENO, &key, 1);
                return bytes_read == 1 ? key : '\0';
            }

            void restore() {
                if (!enabled_) {
                    return;
                }

                tcsetattr(STDIN_FILENO, TCSANOW, &original_terminal_);
                enabled_ = false;
            }

        private:
            bool enabled_ = false;
            termios original_terminal_{};
    };

    void request_training_stop(int signal) {
        requested_stop_signal = signal;
    }

    std::string format_integer(long long value) {
        std::string digits = std::to_string(value);

        for (int position = static_cast<int>(digits.size()) - 3; position > 0; position -= 3) {
            digits.insert(static_cast<size_t>(position), ",");
        }

        return digits;
    }

    std::string format_double(double value, int precision) {
        std::ostringstream output;
        output << std::fixed << std::setprecision(precision) << value;
        return output.str();
    }

    std::string format_architecture(const std::vector<int>& architecture) {
        std::ostringstream output;

        for (size_t i = 0; i < architecture.size(); i++) {
            if (i > 0) {
                output << " -> ";
            }

            output << architecture[i];
        }

        return output.str();
    }

    std::string format_duration(double seconds) {
        if (!std::isfinite(seconds) || seconds < 0.0) {
            return "estimating";
        }

        long long total_seconds = static_cast<long long>(seconds + 0.5);
        long long hours = total_seconds / 3600;
        long long minutes = (total_seconds % 3600) / 60;
        long long remaining_seconds = total_seconds % 60;

        std::ostringstream output;
        if (hours > 0) {
            output << hours << "h ";
        }

        output << std::setw(2) << std::setfill('0') << minutes << "m "
               << std::setw(2) << std::setfill('0') << remaining_seconds << "s";

        return output.str();
    }

    std::string format_timestamp(std::chrono::system_clock::time_point time_point) {
        std::time_t time = std::chrono::system_clock::to_time_t(time_point);
        std::tm* local_time = std::localtime(&time);

        if (local_time == nullptr) {
            return "unknown";
        }

        std::ostringstream output;
        output << std::put_time(local_time, "%Y-%m-%d %H:%M:%S");
        return output.str();
    }

    std::string progress_bar(double ratio, int width) {
        ratio = std::max(0.0, std::min(1.0, ratio));
        int filled = static_cast<int>(ratio * width + 0.5);
        std::string bar = "[";

        for (int i = 0; i < width; i++) {
            bar += i < filled ? '#' : '.';
        }

        bar += "]";
        return bar;
    }

    class TrainingDisplay {
        public:
            TrainingDisplay(
                int epochs,
                int samples_per_epoch,
                int validation_samples,
                int parameter_count,
                double learning_rate,
                bool controls_enabled,
                const std::string& checkpoint_path
            )
                : epochs_(epochs),
                  samples_per_epoch_(samples_per_epoch),
                  validation_samples_(validation_samples),
                  parameter_count_(parameter_count),
                  learning_rate_(learning_rate),
                  controls_enabled_(controls_enabled),
                  checkpoint_path_(checkpoint_path),
                  total_samples_(static_cast<long long>(epochs) * samples_per_epoch),
                  started_at_(Clock::now()),
                  started_wall_time_(std::chrono::system_clock::now()),
                  last_rendered_at_(started_at_ - std::chrono::milliseconds(1000)) {}

            void render(const TrainingState& state, bool force = false) {
                auto now = Clock::now();
                if (!force && now - last_rendered_at_ < std::chrono::milliseconds(200)) {
                    return;
                }

                last_rendered_at_ = now;
                render_count_++;

                double active_elapsed_seconds = state.active_seconds_before_run
                    + std::chrono::duration<double>(now - started_at_).count();
                double throughput = active_elapsed_seconds > 0.0
                    ? state.samples_completed / active_elapsed_seconds
                    : 0.0;
                double eta_seconds = throughput > 0.0
                    ? (total_samples_ - state.samples_completed) / throughput
                    : std::numeric_limits<double>::infinity();
                double overall_progress = total_samples_ > 0
                    ? static_cast<double>(state.samples_completed) / total_samples_
                    : 1.0;
                double epoch_progress = samples_per_epoch_ > 0
                    ? static_cast<double>(state.samples_completed_in_epoch) / samples_per_epoch_
                    : 1.0;
                double running_loss = state.samples_completed_in_epoch > 0
                    ? state.epoch_loss / state.samples_completed_in_epoch
                    : 0.0;
                double running_accuracy = state.samples_completed_in_epoch > 0
                    ? static_cast<double>(state.epoch_correct) * 100.0 / state.samples_completed_in_epoch
                    : 0.0;
                double timed_samples = std::max(1LL, state.samples_completed);
                char spinner = spinner_frames_[render_count_ % spinner_frames_.size()];

                std::cout << "\033[2J\033[H";
                std::cout << "MNIST training " << spinner << "\n";
                std::cout << "Started: " << format_timestamp(started_wall_time_)
                          << "   Updated: " << format_timestamp(std::chrono::system_clock::now()) << "\n\n";
                std::cout << "Controls: "
                          << (controls_enabled_
                                ? "p/space save checkpoint and params, Ctrl+C stop and save"
                                : "not available because stdin is not a terminal")
                          << "\n";
                std::cout << "Checkpoint: " << checkpoint_path_ << "\n";
                std::cout << "Status: "
                          << (state.resumed_from_checkpoint ? "resumed from checkpoint" : "running")
                          << "\n";
                if (!state.notice.empty()) {
                    std::cout << "Notice: " << state.notice << "\n";
                }
                std::cout << "\n";

                std::cout << "Epoch   " << state.epoch << "/" << epochs_
                          << "   done " << format_integer(state.samples_completed_in_epoch)
                          << "/" << format_integer(samples_per_epoch_) << "\n";
                std::cout << progress_bar(epoch_progress, 40)
                          << " " << format_double(epoch_progress * 100.0, 1) << "% this epoch\n\n";

                std::cout << "Total   done " << format_integer(state.samples_completed)
                          << "/" << format_integer(total_samples_) << " samples\n";
                std::cout << progress_bar(overall_progress, 40)
                          << " " << format_double(overall_progress * 100.0, 1) << "% overall\n\n";

                std::cout << "Phase: " << state.phase
                          << "   parameters: " << format_integer(parameter_count_)
                          << "   learning rate: " << learning_rate_ << "\n";
                std::cout << "Epoch loss: " << format_double(running_loss, 6)
                          << "   accuracy: " << format_double(running_accuracy, 2) << "%\n";
                if (state.has_last_validation) {
                    std::cout << "Latest validation: epoch " << state.last_validation_epoch
                              << " | loss " << format_double(state.last_validation.loss, 6)
                              << " | accuracy "
                              << format_double(state.last_validation.accuracy * 100.0, 2) << "%"
                              << " | " << format_integer(state.last_validation.samples)
                              << " samples"
                              << " | " << format_duration(state.last_validation.seconds) << "\n";
                } else if (validation_samples_ > 0) {
                    std::cout << "Latest validation: waiting for first completed epoch over "
                              << format_integer(validation_samples_) << " samples\n";
                }

                if (state.last_label >= 0) {
                    std::cout << "Last sample: label " << state.last_label
                              << ", predicted " << state.last_prediction
                              << (state.last_prediction == state.last_label ? " (correct)" : " (miss)")
                              << ", loss " << format_double(state.last_loss, 6) << "\n";
                } else {
                    std::cout << "Last sample: waiting for first prediction\n";
                }

                std::cout << "\n";
                std::cout << "Elapsed: " << format_duration(active_elapsed_seconds)
                          << "   ETA: " << format_duration(eta_seconds)
                          << "   speed: " << format_double(throughput, 2) << " samples/sec\n";
                std::cout << "Avg phase time/sample: forward "
                          << format_double(state.timers.forward_seconds * 1000.0 / timed_samples, 2) << "ms"
                          << " | loss " << format_double(state.timers.loss_seconds * 1000.0 / timed_samples, 2) << "ms"
                          << " | backward " << format_double(state.timers.backward_seconds * 1000.0 / timed_samples, 2) << "ms"
                          << " | update " << format_double(state.timers.update_seconds * 1000.0 / timed_samples, 2) << "ms\n";

                if (!state.completed_epochs.empty()) {
                    std::cout << "\nCompleted epochs:\n";
                    size_t first_epoch = state.completed_epochs.size() > 5
                        ? state.completed_epochs.size() - 5
                        : 0;

                    for (size_t i = first_epoch; i < state.completed_epochs.size(); i++) {
                        const EpochSummary& summary = state.completed_epochs[i];
                        std::cout << "  " << summary.epoch
                                  << " | loss " << format_double(summary.loss, 6)
                                  << " | accuracy " << format_double(summary.accuracy * 100.0, 2) << "%"
                                  << " | " << format_duration(summary.seconds) << "\n";
                    }
                }

                std::cout << std::flush;
            }

        private:
            const std::string spinner_frames_ = "|/-\\";
            int epochs_;
            int samples_per_epoch_;
            int validation_samples_;
            int parameter_count_;
            double learning_rate_;
            bool controls_enabled_;
            std::string checkpoint_path_;
            long long total_samples_;
            Clock::time_point started_at_;
            std::chrono::system_clock::time_point started_wall_time_;
            Clock::time_point last_rendered_at_;
            size_t render_count_ = 0;
    };

    bool is_checkpoint_key(char key) {
        return key == 'p' || key == 'P' || key == ' ';
    }

    bool checkpoint_requested(TerminalControls& controls) {
        return controls.enabled() && is_checkpoint_key(controls.read_key());
    }

    bool stop_requested() {
        return requested_stop_signal != 0;
    }
}

std::vector<std::shared_ptr<Value>> pixels_to_values(const std::vector<double>& pixels) {
    std::vector<std::shared_ptr<Value>> values;
    values.reserve(pixels.size());

    for (double pixel : pixels) {
        values.push_back(Value::create(pixel));
    }

    return values;
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

EvaluationResult evaluate_model(
    MLP& model,
    const std::vector<MnistSample>& samples
) {
    EvaluationResult result;
    result.samples = static_cast<int>(samples.size());

    if (samples.empty()) {
        return result;
    }

    auto started_at = Clock::now();
    double total_loss = 0.0;
    int correct_count = 0;

    for (const MnistSample& sample : samples) {
        auto input_values = pixels_to_values(sample.pixels);
        auto logits = model(input_values);
        auto loss = softmax_cross_entropy_loss(logits, sample.label);
        int prediction = predicted_label(logits);

        total_loss += loss->data();
        if (prediction == sample.label) {
            correct_count++;
        }
    }

    result.loss = total_loss / result.samples;
    result.accuracy = static_cast<double>(correct_count) / result.samples;
    result.seconds = std::chrono::duration<double>(Clock::now() - started_at).count();
    return result;
}

void save_parameters(
    const std::string& path,
    const std::vector<int>& architecture,
    const std::vector<std::shared_ptr<Value>>& parameters
) {
    std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

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

void expect_tag(std::istream& input, const std::string& expected_tag) {
    std::string actual_tag;
    input >> actual_tag;

    if (actual_tag != expected_tag) {
        throw std::runtime_error(
            "Invalid checkpoint format: expected '" + expected_tag
                + "', got '" + actual_tag + "'"
        );
    }
}

bool checkpoint_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

void remove_checkpoint(const std::string& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
}

void apply_parameter_values(
    const std::vector<std::shared_ptr<Value>>& parameters,
    const std::vector<double>& values
) {
    if (parameters.size() != values.size()) {
        throw std::runtime_error("Checkpoint parameter count does not match the model");
    }

    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i]->set_data(values[i]);
        parameters[i]->set_grad(0.0);
    }
}

CheckpointData load_checkpoint(const std::string& path) {
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error(std::string("Could not open checkpoint file: ") + path);
    }

    CheckpointData checkpoint;
    std::string magic;
    int version = 0;
    file >> magic >> version;

    if (magic != CHECKPOINT_MAGIC || version != CHECKPOINT_VERSION) {
        throw std::runtime_error("Unsupported checkpoint file format");
    }

    expect_tag(file, "epochs");
    file >> checkpoint.epochs;
    expect_tag(file, "training_limit");
    file >> checkpoint.training_limit;
    expect_tag(file, "learning_rate");
    file >> checkpoint.learning_rate;
    expect_tag(file, "epoch_index");
    file >> checkpoint.epoch_index;
    expect_tag(file, "next_sample_position");
    file >> checkpoint.next_sample_position;
    expect_tag(file, "samples_completed");
    file >> checkpoint.samples_completed;
    expect_tag(file, "current_epoch_loss");
    file >> checkpoint.current_epoch_loss;
    expect_tag(file, "current_epoch_correct");
    file >> checkpoint.current_epoch_correct;
    expect_tag(file, "current_epoch_seconds");
    file >> checkpoint.current_epoch_seconds;
    expect_tag(file, "active_seconds");
    file >> checkpoint.active_seconds;
    expect_tag(file, "last_sample");
    file >> checkpoint.last_label >> checkpoint.last_prediction >> checkpoint.last_loss;
    expect_tag(file, "phase_timers");
    file >> checkpoint.timers.forward_seconds
         >> checkpoint.timers.loss_seconds
         >> checkpoint.timers.backward_seconds
         >> checkpoint.timers.update_seconds;

    expect_tag(file, "architecture");
    size_t architecture_size = 0;
    file >> architecture_size;
    checkpoint.architecture.resize(architecture_size);
    for (int& size : checkpoint.architecture) {
        file >> size;
    }

    expect_tag(file, "parameters");
    size_t parameter_count = 0;
    file >> parameter_count;
    checkpoint.parameters.resize(parameter_count);
    for (double& parameter : checkpoint.parameters) {
        file >> parameter;
    }

    expect_tag(file, "rng");
    file >> checkpoint.rng;

    expect_tag(file, "sample_indices");
    size_t sample_index_count = 0;
    file >> sample_index_count;
    checkpoint.sample_indices.resize(sample_index_count);
    for (int& sample_index : checkpoint.sample_indices) {
        file >> sample_index;
    }

    expect_tag(file, "completed_epochs");
    size_t completed_epoch_count = 0;
    file >> completed_epoch_count;
    checkpoint.completed_epochs.resize(completed_epoch_count);
    for (EpochSummary& summary : checkpoint.completed_epochs) {
        file >> summary.epoch >> summary.loss >> summary.accuracy >> summary.seconds;
    }

    if (!file) {
        throw std::runtime_error("Checkpoint file is incomplete or corrupt");
    }

    return checkpoint;
}

void validate_checkpoint(
    const CheckpointData& checkpoint,
    int epochs,
    int training_limit,
    double learning_rate,
    size_t parameter_count
) {
    if (checkpoint.architecture != DIGITS_ARCHITECTURE) {
        throw std::runtime_error("Checkpoint architecture does not match this trainer");
    }
    if (checkpoint.epochs != epochs
        || checkpoint.training_limit != training_limit
        || std::fabs(checkpoint.learning_rate - learning_rate) > 1e-12) {
        throw std::runtime_error(
            "Checkpoint settings do not match this run. Use the same arguments to resume, "
            "or pass --fresh to start over."
        );
    }
    if (checkpoint.epoch_index < 0 || checkpoint.epoch_index >= epochs) {
        throw std::runtime_error("Checkpoint epoch is out of range");
    }
    if (checkpoint.next_sample_position < 0
        || checkpoint.next_sample_position > training_limit) {
        throw std::runtime_error("Checkpoint sample position is out of range");
    }
    if (checkpoint.sample_indices.size() != static_cast<size_t>(training_limit)) {
        throw std::runtime_error("Checkpoint sample order does not match the training limit");
    }
    if (checkpoint.parameters.size() != parameter_count) {
        throw std::runtime_error("Checkpoint parameter count does not match this model");
    }
}

void save_checkpoint(
    const std::string& path,
    const std::vector<std::shared_ptr<Value>>& parameters,
    int epochs,
    int training_limit,
    double learning_rate,
    int epoch_index,
    int next_sample_position,
    double current_epoch_seconds,
    double active_seconds,
    const TrainingState& state,
    const std::mt19937& rng,
    const std::vector<int>& sample_indices
) {
    std::filesystem::path checkpoint_path(path);
    if (checkpoint_path.has_parent_path()) {
        std::filesystem::create_directories(checkpoint_path.parent_path());
    }

    std::string temporary_path = path + ".tmp";
    std::ofstream file(temporary_path);

    if (!file) {
        throw std::runtime_error(std::string("Could not open checkpoint file: ") + temporary_path);
    }

    file << std::setprecision(17);
    file << CHECKPOINT_MAGIC << " " << CHECKPOINT_VERSION << "\n";
    file << "epochs " << epochs << "\n";
    file << "training_limit " << training_limit << "\n";
    file << "learning_rate " << learning_rate << "\n";
    file << "epoch_index " << epoch_index << "\n";
    file << "next_sample_position " << next_sample_position << "\n";
    file << "samples_completed " << state.samples_completed << "\n";
    file << "current_epoch_loss " << state.epoch_loss << "\n";
    file << "current_epoch_correct " << state.epoch_correct << "\n";
    file << "current_epoch_seconds " << current_epoch_seconds << "\n";
    file << "active_seconds " << active_seconds << "\n";
    file << "last_sample " << state.last_label << " "
         << state.last_prediction << " " << state.last_loss << "\n";
    file << "phase_timers "
         << state.timers.forward_seconds << " "
         << state.timers.loss_seconds << " "
         << state.timers.backward_seconds << " "
         << state.timers.update_seconds << "\n";

    file << "architecture " << DIGITS_ARCHITECTURE.size();
    for (int size : DIGITS_ARCHITECTURE) {
        file << " " << size;
    }
    file << "\n";

    file << "parameters " << parameters.size() << "\n";
    for (const auto& parameter : parameters) {
        file << parameter->data() << "\n";
    }

    file << "rng\n" << rng << "\n";

    file << "sample_indices " << sample_indices.size() << "\n";
    for (int sample_index : sample_indices) {
        file << sample_index << " ";
    }
    file << "\n";

    file << "completed_epochs " << state.completed_epochs.size() << "\n";
    for (const EpochSummary& summary : state.completed_epochs) {
        file << summary.epoch << " "
             << summary.loss << " "
             << summary.accuracy << " "
             << summary.seconds << "\n";
    }

    file.close();
    if (!file) {
        throw std::runtime_error(std::string("Could not write checkpoint file: ") + temporary_path);
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
    if (rename_error) {
        std::filesystem::remove(path);
        std::filesystem::rename(temporary_path, path);
    }
}

TrainingResult train_model(
    MLP& model,
    const std::vector<MnistSample>& samples,
    const std::vector<MnistSample>& validation_samples,
    int epochs,
    int training_limit,
    double learning_rate,
    const std::string& checkpoint_path,
    bool fresh_start
) {
    requested_stop_signal = 0;

    if (epochs < 1) {
        throw std::runtime_error("Epoch count must be at least 1");
    }

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
    TrainingState state;
    int start_epoch = 0;
    int start_position = 0;
    double current_epoch_seconds_before_run = 0.0;

    if (!fresh_start && checkpoint_exists(checkpoint_path)) {
        try {
            CheckpointData checkpoint = load_checkpoint(checkpoint_path);
            validate_checkpoint(
                checkpoint,
                epochs,
                training_limit,
                learning_rate,
                parameters.size()
            );

            apply_parameter_values(parameters, checkpoint.parameters);
            sample_indices = checkpoint.sample_indices;
            rng = checkpoint.rng;
            start_epoch = checkpoint.epoch_index;
            start_position = checkpoint.next_sample_position;
            current_epoch_seconds_before_run = checkpoint.current_epoch_seconds;

            state.epoch = checkpoint.epoch_index + 1;
            state.samples_completed_in_epoch = checkpoint.next_sample_position;
            state.samples_completed = checkpoint.samples_completed;
            state.phase = "resumed from checkpoint";
            state.epoch_loss = checkpoint.current_epoch_loss;
            state.epoch_correct = checkpoint.current_epoch_correct;
            state.last_loss = checkpoint.last_loss;
            state.last_label = checkpoint.last_label;
            state.last_prediction = checkpoint.last_prediction;
            state.resumed_from_checkpoint = true;
            state.active_seconds_before_run = checkpoint.active_seconds;
            state.timers = checkpoint.timers;
            state.completed_epochs = checkpoint.completed_epochs;
        } catch (const std::exception& error) {
            state.notice = "ignored existing checkpoint: " + std::string(error.what());
        }
    }

    TerminalControls controls;
    auto run_started_at = Clock::now();
    TrainingDisplay display(
        epochs,
        training_limit,
        static_cast<int>(validation_samples.size()),
        static_cast<int>(parameters.size()),
        learning_rate,
        controls.enabled(),
        checkpoint_path
    );

    display.render(state, true);

    auto save_exit_checkpoint = [&](int epoch_index,
                                    int next_sample_position,
                                    double current_epoch_seconds,
                                    bool stopped) -> TrainingResult {
        state.phase = stopped ? "saving stop checkpoint" : "saving checkpoint";
        display.render(state, true);
        save_checkpoint(
            checkpoint_path,
            parameters,
            epochs,
            training_limit,
            learning_rate,
            epoch_index,
            next_sample_position,
            current_epoch_seconds,
            state.active_seconds_before_run
                + std::chrono::duration<double>(Clock::now() - run_started_at).count(),
            state,
            rng,
            sample_indices
        );
        state.phase = stopped ? "stop checkpoint saved" : "checkpoint saved";
        display.render(state, true);
        return {false, true, stopped, checkpoint_path};
    };

    for (int epoch = start_epoch; epoch < epochs; epoch++) {
        auto epoch_started_at = Clock::now();
        int epoch_start_position = 0;

        if (state.resumed_from_checkpoint && epoch == start_epoch) {
            epoch_start_position = start_position;
            state.phase = "resuming epoch";
            display.render(state, true);
        } else {
            current_epoch_seconds_before_run = 0.0;
            state.epoch = epoch + 1;
            state.samples_completed_in_epoch = 0;
            state.epoch_loss = 0.0;
            state.epoch_correct = 0;
            state.phase = "shuffling samples";
            display.render(state, true);

            std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
        }

        bool pause_requested = checkpoint_requested(controls);
        bool stopping = stop_requested();
        if (pause_requested || stopping) {
            return save_exit_checkpoint(
                epoch,
                epoch_start_position,
                current_epoch_seconds_before_run
                    + std::chrono::duration<double>(Clock::now() - epoch_started_at).count(),
                stopping
            );
        }

        double total_loss = state.epoch_loss;
        int correct_count = state.epoch_correct;

        for (int position = epoch_start_position; position < training_limit; position++) {
            int sample_index = sample_indices[position];
            const MnistSample& sample = samples[sample_index];

            state.phase = "forward";
            display.render(state);
            auto phase_started_at = Clock::now();

            auto input_values = pixels_to_values(sample.pixels);
            auto logits = model(input_values);

            state.timers.forward_seconds += std::chrono::duration<double>(
                Clock::now() - phase_started_at
            ).count();

            state.phase = "loss + prediction";
            display.render(state);
            phase_started_at = Clock::now();

            auto loss = softmax_cross_entropy_loss(logits, sample.label);
            int prediction = predicted_label(logits);

            total_loss += loss->data();

            if (prediction == sample.label) {
                correct_count++;
            }

            state.epoch_loss = total_loss;
            state.epoch_correct = correct_count;
            state.last_loss = loss->data();
            state.last_label = sample.label;
            state.last_prediction = prediction;
            state.timers.loss_seconds += std::chrono::duration<double>(
                Clock::now() - phase_started_at
            ).count();

            state.phase = "backward";
            display.render(state);
            phase_started_at = Clock::now();

            zero_gradients(parameters);
            loss->backward();

            state.timers.backward_seconds += std::chrono::duration<double>(
                Clock::now() - phase_started_at
            ).count();

            state.phase = "parameter update";
            display.render(state);
            phase_started_at = Clock::now();

            update_parameters(parameters, learning_rate);

            state.timers.update_seconds += std::chrono::duration<double>(
                Clock::now() - phase_started_at
            ).count();
            state.samples_completed_in_epoch++;
            state.samples_completed++;
            state.phase = "sample complete";
            display.render(state);

            pause_requested = checkpoint_requested(controls);
            stopping = stop_requested();
            if (pause_requested || stopping) {
                return save_exit_checkpoint(
                    epoch,
                    position + 1,
                    current_epoch_seconds_before_run
                        + std::chrono::duration<double>(Clock::now() - epoch_started_at).count(),
                    stopping
                );
            }
        }

        double average_loss = total_loss / training_limit;
        double accuracy = static_cast<double>(correct_count) / training_limit;
        double epoch_seconds = current_epoch_seconds_before_run
            + std::chrono::duration<double>(Clock::now() - epoch_started_at).count();

        state.completed_epochs.push_back({epoch + 1, average_loss, accuracy, epoch_seconds});
        state.samples_completed_in_epoch = training_limit;

        if (!validation_samples.empty()) {
            state.phase = "validating";
            display.render(state, true);
            state.last_validation = evaluate_model(model, validation_samples);
            state.last_validation_epoch = epoch + 1;
            state.has_last_validation = true;
        }

        state.phase = "epoch complete";
        display.render(state, true);
        state.resumed_from_checkpoint = false;
    }

    state.phase = "training complete";
    display.render(state, true);
    return {true, false, stop_requested(), checkpoint_path};
}

int main(int argc, char* argv[]) {
    try {
        TrainingOptions options;
        std::vector<std::string> positional_arguments;

        for (int i = 1; i < argc; i++) {
            std::string argument = argv[i];

            if (argument == "--fresh") {
                options.fresh_start = true;
            } else if (argument == "--checkpoint") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--checkpoint requires a path");
                }
                options.checkpoint_path = argv[++i];
            } else if (argument == "--validation-limit") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--validation-limit requires a sample count");
                }
                options.validation_limit = std::stoi(argv[++i]);
            } else if (argument == "--test-limit") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--test-limit requires a sample count");
                }
                options.test_limit = std::stoi(argv[++i]);
            } else if (argument.rfind("--", 0) == 0) {
                throw std::runtime_error(std::string("Unknown option: ") + argument);
            } else {
                positional_arguments.push_back(argument);
            }
        }

        if (positional_arguments.size() > 3) {
            throw std::runtime_error(
                "Usage: train_mnist [training_limit] [epochs] [learning_rate] "
                "[--fresh] [--checkpoint path] [--validation-limit samples] "
                "[--test-limit samples]"
            );
        }

        if (positional_arguments.size() > 0) {
            options.training_limit = std::stoi(positional_arguments[0]);
        }
        if (positional_arguments.size() > 1) {
            options.epochs = std::stoi(positional_arguments[1]);
        }
        if (positional_arguments.size() > 2) {
            options.learning_rate = std::stod(positional_arguments[2]);
        }

        if (options.validation_limit < 0 || options.test_limit < 0) {
            throw std::runtime_error("Validation and test limits cannot be negative");
        }

        auto training_samples = load_mnist_dataset(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
            options.training_limit
        );

        options.training_limit = std::min(
            options.training_limit,
            static_cast<int>(training_samples.size())
        );

        std::vector<MnistSample> validation_samples;
        std::vector<MnistSample> test_samples;
        int evaluation_limit = options.validation_limit + options.test_limit;

        if (evaluation_limit > 0) {
            auto evaluation_samples = load_mnist_dataset(
                "data/mnist/t10k-images-idx3-ubyte",
                "data/mnist/t10k-labels-idx1-ubyte",
                evaluation_limit
            );

            int validation_count = std::min(
                options.validation_limit,
                static_cast<int>(evaluation_samples.size())
            );
            validation_samples.assign(
                evaluation_samples.begin(),
                evaluation_samples.begin() + validation_count
            );

            int remaining_evaluation_samples =
                static_cast<int>(evaluation_samples.size()) - validation_count;
            int test_count = std::min(options.test_limit, remaining_evaluation_samples);
            test_samples.assign(
                evaluation_samples.begin() + validation_count,
                evaluation_samples.begin() + validation_count + test_count
            );
        }

        std::cout << "Loaded " << training_samples.size() << " MNIST training samples\n";
        std::cout << "Loaded " << validation_samples.size() << " validation samples and "
                  << test_samples.size() << " test samples from the held-out MNIST set\n"
                  << std::flush;

        std::vector<int> layer_outputs(
            DIGITS_ARCHITECTURE.begin() + 1,
            DIGITS_ARCHITECTURE.end()
        );
        MLP model(DIGITS_ARCHITECTURE.front(), layer_outputs, Activation::Linear);
        auto parameters = model.parameters();

        std::cout << "Created MLP with architecture "
                  << format_architecture(DIGITS_ARCHITECTURE) << "\n";
        std::cout << "Parameter count: " << parameters.size() << "\n";

        std::cout << "Training on " << options.training_limit
                  << " samples for " << options.epochs
                  << " epochs with learning rate " << options.learning_rate << "\n";
        std::cout << "Validation: " << validation_samples.size()
                  << " samples after each epoch"
                  << "   Final test: " << test_samples.size() << " samples\n";
        std::cout << "Checkpoint path: " << options.checkpoint_path << "\n";
        if (options.fresh_start && checkpoint_exists(options.checkpoint_path)) {
            std::cout << "Ignoring existing checkpoint because --fresh was provided\n";
        }
        std::cout << std::flush;

        TrainingResult result = train_model(
            model,
            training_samples,
            validation_samples,
            options.epochs,
            options.training_limit,
            options.learning_rate,
            options.checkpoint_path,
            options.fresh_start
        );

        if (result.checkpointed) {
            save_parameters(DEFAULT_MODEL_PATH, DIGITS_ARCHITECTURE, parameters);
            std::cout << "\n"
                      << (result.stopped ? "Stopped training." : "Paused training.")
                      << "\n";
            std::cout << "Saved checkpoint to " << result.checkpoint_path << "\n";
            std::cout << "Saved current model parameters to " << DEFAULT_MODEL_PATH << "\n";
            std::cout << "Run the same command again to continue from that checkpoint.\n";
            std::cout << "Use --fresh if you want to ignore it and start over.\n";
            return 0;
        }

        if (!result.stopped && !test_samples.empty()) {
            std::cout << "\nEvaluating final test set...\n" << std::flush;
            EvaluationResult test_result = evaluate_model(model, test_samples);
            std::cout << "Test loss: " << format_double(test_result.loss, 6)
                      << "   accuracy: "
                      << format_double(test_result.accuracy * 100.0, 2) << "%"
                      << "   samples: " << format_integer(test_result.samples)
                      << "   time: " << format_duration(test_result.seconds) << "\n";
        } else if (result.stopped) {
            std::cout << "\nStopped training before final test evaluation.\n";
        }

        save_parameters(DEFAULT_MODEL_PATH, DIGITS_ARCHITECTURE, parameters);
        remove_checkpoint(options.checkpoint_path);
        std::cout << "Saved model to " << DEFAULT_MODEL_PATH << "\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
