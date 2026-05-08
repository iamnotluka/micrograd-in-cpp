#include "value.h"
#include "graph.h"
#include <mlp.h>
#include <iostream>
#include <vector>

int main() {
    MLP mlp(3, {4, 4, 1});

    std::vector<std::vector<std::shared_ptr<Value>>> xs = {
        {Value::create(2.0), Value::create(3.0), Value::create(-1.0)},
        {Value::create(3.0), Value::create(-1.0), Value::create(0.5)},
        {Value::create(0.5), Value::create(1.0), Value::create(1.0)},
        {Value::create(-1.0), Value::create(1.0), Value::create(-1.0)}
    };

    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    double step_size = 0.05;

    int step_count = 0;

    while (true) {
        std::cout << "\033[2J\033[H";

        std::vector<std::shared_ptr<Value>> predictions;
        for (auto& x : xs) {
            predictions.push_back(mlp(x)[0]);
        }

        auto loss = Value::create(0.0);
        for (int i = 0; i < predictions.size(); i++) {
            auto diff = predictions[i] - ys[i];
            loss = loss + diff->pow(diff, 2.0);
        }

        std::cout << "Step " << step_count << std::endl;
        std::cout << "\nLoss: " << loss->data() << std::endl;
        for (int i = 0; i < predictions.size(); i++) {
            std::cout << "  pred[" << i << "]: " << predictions[i]->data()
                      << "\t(target: " << ys[i] << ")" << std::endl;
        }

        for (auto& p : mlp.parameters()) {
            p->set_grad(0.0);
        }

        loss->backward();

        for (auto& p : mlp.parameters()) {
            p->set_data(p->data() - step_size * p->grad());
        }
        
        step_count++;
        
        std::cout << "\nPress Enter for next step (q to quit)..." << std::endl;
        if (std::cin.get() == 'q') break;
        
    }

    return 0;
}
