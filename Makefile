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

test:
	clang++ -std=c++17 -I engine/include engine/tests/test_gradients.cpp -o test_gradients
	./test_gradients