# micrograd in CPP

Inspired by Andrej Karpathy's lecture: https://www.youtube.com/watch?v=VMj-3S1tku0
Original micrograd repo: https://github.com/karpathy/micrograd

This repo exists to:

- relearn and solidify my understanding of basic ML concepts from the ground up
- get better at C++ while doing it
- end up with something useful for educational purposes, mine first, maybe someone else's later

The plan is to build a working autograd engine that implements reverse-mode autodiff over tensors, a small neural network library on top of it, and a live demo of a trained NN: an interactive canvas where you draw a digit and tells you what it sees in real time.

The static webapp ships as vanilla HTML plus a WASM module with the trained weights baked in, totalling a few hundred KB at most.

The browser fetches it once on page load and runs the WASM locally.

As I build, I'll most probably hit concepts I'm meeting fresh and concepts I'm meeting again with more context. I learn best by writing things down and explaining them, so the deeper notes will live on my blog: www.zoricl.io

Live demo: TODO

### Usability

#### Visualising expressions

Change expression in `main.cpp` by adjusting the values and expression using the values. The output layer of an MLP is itself just a Value expression, so you can feed it straight into the graph.

`make graph` will compile everything, run it, and open a visual graph of the computation.

#### Training loop

`manual_loss.cpp` has a bare-bones training loop you can step through by hand. It runs a forward pass, computes the loss, backpropagates, and nudges the parameters. Each time you press Enter you see the updated loss and predictions. Press `q` to quit.