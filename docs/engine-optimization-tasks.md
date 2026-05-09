# Engine Optimization Tasks

These are performance-oriented tasks for the C++ micrograd engine. They range from easy wins to deeper rewrites. The current MNIST trainer is intentionally educational and scalar-heavy, so the biggest gains will come from reducing graph allocation, pointer overhead, and per-sample work.

Current baseline to keep in mind:

- MNIST training is one sample at a time.
- Every operation creates a `Value` node.
- Nodes are heap allocated through `std::shared_ptr`.
- Backward builds a topology with recursive DFS and `std::set`.
- Each node stores a `std::function` closure for local backward.
- Neuron forward is a scalar loop over 784 input values.

## Task 1: Add A Repeatable Benchmark Harness

Priority: P0
Difficulty: Low
Expected impact: Makes every later optimization measurable.

Goal:

Create a small benchmark command that can time forward, backward, and parameter update separately without the terminal dashboard.

Implementation notes:

- Add `engine/benchmarks/bench_mnist_step.cpp` or similar.
- Run a fixed number of MNIST samples with a fixed seed.
- Report samples/sec, forward ms/sample, backward ms/sample, update ms/sample, and peak graph node count if available.
- Add a Makefile target, for example `make bench-mnist`.
- Compile benchmarks with `-O3` in addition to the normal debug-like build.

Acceptance checks:

- Benchmark runs without the dashboard.
- Output is stable enough to compare before/after changes.
- README or this doc records the baseline number.

## Task 2: Build With Optimization Flags For Training

Priority: P0
Difficulty: Low
Expected impact: Medium. This is the cheapest speedup.

Goal:

Compile `train_mnist` with optimization enabled.

Implementation notes:

- Change the Makefile train target from plain `clang++ -std=c++17 ...` to include `-O3 -DNDEBUG`.
- Consider adding separate targets:
  - `make train-mnist-debug`
  - `make train-mnist-release`
- Keep tests easy to run in debug mode.

Acceptance checks:

- `make train-mnist` still works.
- `make test` still passes.
- Benchmark shows a clear before/after improvement.

## Task 3: Stop Copying Vectors On Forward Calls

Priority: P1
Difficulty: Low
Expected impact: Low to medium.

Goal:

Remove avoidable vector copies in MLP, Layer, and Neuron forward paths.

Current issue:

- `MLP::operator()` takes `std::vector<std::shared_ptr<Value>> input_values` by value.
- `Layer::operator()` takes by value.
- `Neuron::operator()` takes by value.

Implementation notes:

- Prefer `const std::vector<std::shared_ptr<Value>>&` for inputs.
- Return output vectors by value where ownership of the new vector is needed.
- Reserve output vector capacity in `Layer::operator()`.
- Audit `parameters()` methods for unnecessary temporary copies.

Acceptance checks:

- Gradient tests still pass.
- Benchmark shows no regression.
- Public usage remains simple.

## Task 4: Replace `std::set` In Backward Topology Build

Priority: P1
Difficulty: Medium
Expected impact: Medium.

Goal:

Make topology construction cheaper.

Current issue:

`Value::backward()` uses `std::set<Value*> visited`, which is tree-based and does allocation/comparison work per node.

Implementation notes:

- Replace with `std::unordered_set<Value*>`.
- Reserve a reasonable capacity if graph size can be estimated.
- Consider an explicit iterative DFS to avoid recursive `std::function` overhead.
- Avoid creating the aliasing `std::shared_ptr<Value>(this, [](Value*){})` if a raw-pointer traversal can be used safely inside `backward()`.

Acceptance checks:

- Existing gradient tests still pass.
- Deep computation graphs do not overflow recursion if iterative DFS is implemented.
- Benchmark shows backward topology build improvement.

## Task 5: Remove `std::function` From Every Value Node

Priority: P1
Difficulty: High
Expected impact: High.

Goal:

Replace per-node captured lambdas with a compact operation enum and direct backward switch.

Current issue:

Every operation stores a `std::function<void()>`, usually with captured `shared_ptr`s and an output pointer. This is flexible but expensive.

Implementation notes:

- Add an enum like:

```cpp
enum class Op { None, Add, Sub, Mul, Div, Pow, Tanh, Exp };
```

- Store raw parent pointers or compact parent references in the node.
- Store operation-specific constants, for example exponent for `pow`.
- Implement `Value::local_backward()` as a `switch (op_)`.
- Keep labels/debug information optional.

Acceptance checks:

- All gradient tests pass.
- Graph rendering still has enough op metadata.
- Benchmark shows reduced forward allocation/closure overhead and faster backward.

## Task 6: Introduce A Value Arena For Per-Sample Graph Nodes

Priority: P1
Difficulty: High
Expected impact: Very high.

Goal:

Replace thousands of tiny heap allocations per sample with arena allocation.

Current issue:

Every intermediate `Value` is allocated individually with `std::make_shared<Value>`. MNIST creates a large graph for every sample, then throws it away.

Implementation notes:

- Add a `ValueArena` that owns nodes for one training step.
- Allocate intermediate graph nodes from the arena.
- Reset the arena after each sample.
- Keep model parameters outside the arena because they persist across samples.
- Start with a conservative API, for example `arena.create(data, parents, op)`.

Acceptance checks:

- No dangling parents after arena reset.
- Parameters survive across samples.
- Gradient tests pass with both normal and arena-backed creation if both modes exist.
- Benchmark shows much less allocation overhead.

## Task 7: Split Parameter Values From Graph Nodes

Priority: P2
Difficulty: Very high
Expected impact: Very high.

Goal:

Stop treating every persistent weight as the same object type as every temporary computation node.

Current issue:

Model parameters and temporary computation nodes both use `Value`. This is simple but makes memory ownership and graph lifetime expensive.

Implementation notes:

- Introduce a lighter `Parameter` type storing only `data` and `grad`.
- Let graph nodes reference parameters without owning them.
- Keep parameter update loops dense and cache-friendly.
- This pairs naturally with a graph arena.

Acceptance checks:

- Training behavior stays numerically equivalent within a small tolerance.
- Parameter serialization still works.
- Benchmark improves memory use and update speed.

## Task 8: Add A Fused Dot Product Operation

Priority: P2
Difficulty: High
Expected impact: Very high for MLPs.

Goal:

Replace hundreds of scalar multiply/add graph nodes per neuron with one fused differentiable dot product node.

Current issue:

Each neuron currently builds:

- 784 multiply nodes
- 784 add nodes
- 1 bias add
- 1 tanh

For MNIST, that cost dominates.

Implementation notes:

- Add an operation that computes `sum(weights[i] * inputs[i]) + bias`.
- During backward:
  - `weight[i].grad += out.grad * input[i].data`
  - `input[i].grad += out.grad * weight[i].data`
  - `bias.grad += out.grad`
- Keep this as a deliberate "fast path" for dense layers while preserving scalar ops for education.

Acceptance checks:

- Add focused gradient tests for fused dot product.
- Compare output and gradients against the existing scalar implementation.
- Benchmark should show a large speedup in MLP training.

## Task 9: Add A Dense Layer Fast Path

Priority: P2
Difficulty: Very high
Expected impact: Extremely high.

Goal:

Implement a layer-level forward/backward path using contiguous arrays instead of one graph node per scalar operation.

Implementation notes:

- Store weights as a flat row-major vector:
  - `weights[output_index * input_count + input_index]`
- Store gradients in matching flat arrays.
- Forward:
  - matrix-vector multiply
  - add bias
  - apply activation
- Backward:
  - compute output activation derivative
  - accumulate weight, bias, and input gradients
- Keep the existing scalar engine for examples and graph visualization.

Acceptance checks:

- Dense layer gradients match finite-difference checks.
- MNIST trainer can opt into dense mode.
- Benchmark shows orders-of-magnitude improvement compared with scalar graph mode.

## Task 10: Add Mini-Batch Training

Priority: P2
Difficulty: High
Expected impact: High, especially after dense layers exist.

Goal:

Train multiple samples per update instead of one sample per update.

Implementation notes:

- Add a batch size argument to `train_mnist`.
- Accumulate gradients over a batch.
- Divide gradients or learning rate by batch size to keep update scale controlled.
- Shuffle once per epoch, then iterate in batches.
- Report batch/sec and sample/sec in the dashboard.

Acceptance checks:

- Batch size 1 matches current behavior.
- Larger batches train without exploding loss.
- Benchmark reports better throughput.

## Task 11: Reuse Input Value Objects

Priority: P2
Difficulty: Medium
Expected impact: Medium.

Goal:

Avoid allocating 784 fresh input `Value` objects for every MNIST sample.

Implementation notes:

- Keep a reusable vector of input nodes.
- Set their `data` for each sample.
- Make sure `grad` and graph parent data are reset correctly.
- This is safest after graph lifetime is clearer, because reused input nodes can accidentally retain old edges if not isolated.

Acceptance checks:

- Gradients for inputs are reset correctly.
- No previous sample graph remains reachable.
- Benchmark shows reduced forward setup time.

## Task 12: Make Debug Metadata Optional

Priority: P3
Difficulty: Medium
Expected impact: Low to medium.

Goal:

Avoid storing operation strings and labels in hot training nodes unless graph visualization needs them.

Current issue:

Every `Value` has `std::string op_` and `std::string label_`. These are useful for graph output, but unnecessary in hot MNIST training.

Implementation notes:

- Use an enum for operation identity.
- Gate labels behind a debug build flag or optional side table.
- Keep graph rendering working in debug/visualization mode.

Acceptance checks:

- Normal examples still print useful values.
- Graph rendering still works when debug metadata is enabled.
- Benchmark shows no regression.

## Task 13: Add Memory And Allocation Instrumentation

Priority: P3
Difficulty: Medium
Expected impact: Indirect, but important for optimization work.

Goal:

Track how many graph nodes and heap allocations a training sample creates.

Implementation notes:

- Add counters to `Value::create`.
- Count operation types.
- Report graph nodes/sample and nodes/sec in benchmark mode.
- Optionally add a custom allocator later.

Acceptance checks:

- Counters can be enabled without noisy output during normal training.
- Benchmark prints useful allocation metrics.
- Optimization PRs can show node/allocation reductions.

## Task 14: Add Finite-Difference Gradient Tests For New Fast Paths

Priority: P0 for any fused/dense rewrite
Difficulty: Medium
Expected impact: Prevents silent training bugs.

Goal:

Before replacing scalar graph code with fused operations, add numerical gradient checks.

Implementation notes:

- Use small deterministic networks.
- Compare analytic gradients with finite differences.
- Test:
  - fused dot product
  - dense layer
  - tanh activation
  - batched loss
- Keep tolerance realistic because floating point differences will appear.

Acceptance checks:

- Tests fail if a sign or scaling factor is wrong.
- Tests are fast enough for `make test`.

## Suggested Order

1. Add benchmarks.
2. Build with `-O3`.
3. Remove avoidable vector copies.
4. Replace `std::set` in topology build.
5. Add allocation/node counters.
6. Remove `std::function` from `Value`.
7. Add arena allocation.
8. Add fused dot product.
9. Add dense layer fast path.
10. Add mini-batching.

The first four tasks keep the current design intact. The later tasks change the shape of the engine and should be done with benchmark numbers and gradient tests in place.
