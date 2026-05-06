build:
	clang++ -std=c++17 -I engine/include engine/src/main.cpp -o main

run:
	@if [ -f main ]; then ./main; else echo "No binary found. Run 'make build' first."; fi

build-run: build run

graph: build
	./main
	dot -Tpng graph.dot -o graph.png
	open graph.png
