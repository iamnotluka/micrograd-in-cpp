TRAIN_MNIST_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
ifeq ($(TRAIN_MNIST_ARGS),)
TRAIN_MNIST_ARGS := 50 2 0.01
endif

PREDICT_MNIST_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
ifeq ($(PREDICT_MNIST_ARGS),)
PREDICT_MNIST_ARGS := 1000 20
endif

DEMO_PORT ?= 8000

build:
	clang++ -std=c++17 -I engine/include engine/src/main.cpp -o main

run:
	@if [ -f main ]; then ./main; else echo "No binary found. Run 'make build' first."; fi

build-run: build run

graph: build
	./main
	dot -Tpng graph.dot -o graph.png
	open graph.png

training-sim:
	clang++ -std=c++17 -I engine/include engine/src/manual_loss.cpp -o manual_loss
	./manual_loss

inspect-mnist:
	clang++ -std=c++17 mnist_digits/inspect_mnist.cpp -o mnist_digits/inspect_mnist
	./mnist_digits/inspect_mnist

train-mnist:
	mkdir -p models
	clang++ -std=c++17 -I engine/include mnist_digits/train_mnist.cpp -o mnist_digits/train_mnist
	./mnist_digits/train_mnist $(TRAIN_MNIST_ARGS)

predict-mnist:
	clang++ -std=c++17 mnist_digits/predict_mnist.cpp -o mnist_digits/predict_mnist
	./mnist_digits/predict_mnist $(PREDICT_MNIST_ARGS)
	
test:
	clang++ -std=c++17 -I engine/include engine/tests/test_gradients.cpp -o test_gradients
	./test_gradients

.PHONY: demo
demo:
	@echo "Serving demo at http://localhost:$(DEMO_PORT)/demo/"
	python3 -m http.server $(DEMO_PORT)

%:
	@:
