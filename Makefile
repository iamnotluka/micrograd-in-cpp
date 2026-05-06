build:
	clang++ -I engine/include engine/src/main.cpp -o main

run:
	@if [ -f main ]; then ./main; else echo "No binary found. Run 'make build' first."; fi

build-run: build run