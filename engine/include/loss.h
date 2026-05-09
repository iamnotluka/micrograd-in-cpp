#pragma once

#include <value.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

inline std::shared_ptr<Value> softmax_cross_entropy_loss(
    const std::vector<std::shared_ptr<Value>>& logits,
    int label
) {
    if (logits.empty()) {
        throw std::invalid_argument("Softmax cross entropy needs at least one logit");
    }

    if (label < 0 || label >= static_cast<int>(logits.size())) {
        throw std::invalid_argument("Label is outside the logit range");
    }

    double max_logit = logits[0]->data();
    for (const auto& logit : logits) {
        max_logit = std::max(max_logit, logit->data());
    }

    std::vector<double> probabilities(logits.size());
    double sum_exp = 0.0;

    for (size_t i = 0; i < logits.size(); i++) {
        probabilities[i] = std::exp(logits[i]->data() - max_logit);
        sum_exp += probabilities[i];
    }

    if (sum_exp <= 0.0 || !std::isfinite(sum_exp)) {
        throw std::runtime_error("Softmax cross entropy produced an invalid normalization term");
    }

    for (double& probability : probabilities) {
        probability /= sum_exp;
    }

    double log_sum_exp = max_logit + std::log(sum_exp);
    auto loss = Value::create(log_sum_exp - logits[static_cast<size_t>(label)]->data(), logits, "softmax_cross_entropy");
    Value* out = loss.get();

    loss->set_local_backward([logits, probabilities, label, out]() {
        for (size_t i = 0; i < logits.size(); i++) {
            double target = i == static_cast<size_t>(label) ? 1.0 : 0.0;
            logits[i]->set_grad(logits[i]->grad() + out->grad() * (probabilities[i] - target));
        }
    });

    return loss;
}
