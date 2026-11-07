# JEPA from Scratch

A **Joint Embedding Predictive Architecture (JEPA)** implemented from scratch in **C++26**, without external machine-learning libraries. It learns to predict the future of a bouncing ball — not by predicting pixels, but by predicting the *representation* of the next frame.

The neural networks, backpropagation, EMA target update, and variance regularization are implemented manually.

## What it predicts

A ball moves in a 10 × 10 box with random starting position and velocity, bouncing off the walls. Each frame is just 2 numbers:

```
[x_normalized, y_normalized]  // both in [0, 1]
```

The model sees 5 past frames and predicts the embedding of the next frame. It never sees the ball as an image, and it never tries to reconstruct pixels.

## Architecture

| Component | Configuration |
| --- | --- |
| Input | 2 numbers per frame, 5 context frames |
| Context encoder | 2 → 128 (ReLU) → 64 (ReLU) → 32 (linear) |
| Target encoder | Same as context encoder, updated by EMA only |
| Predictor | 160 → 128 (ReLU) → 64 (ReLU) → 32 (linear) |
| Embedding size | 32 |
| Context frames | 5 |
| Predictor input | 5 × 32 = 160, the 5 context encodings concatenated |
| Output | 32-dim predicted next-frame encoding |
| Optimizer | SGD, learning rate 0.001 |
| Target update | EMA, decay 0.99 |
| Initialization | He, std-dev sqrt(2 / fan_in) |

## How it works

1. **Simulate:** Generate 100 frames of a ball trajectory. Each epoch starts with a new random position and velocity, with a 50% chance of flipping each velocity sign. Velocity is 0 to 1.0 per step, and it reverses on wall hits.
2. **Encode context:** Encode each of the 5 context frames separately with the context encoder, keeping each frame's activations for backprop. Concatenate the 5 × 32-dim encodings into a 160-dim vector.
3. **Predict:** The predictor maps that 160-dim vector to a 32-dim guess of the next frame's encoding.
4. **Encode target:** The target encoder — a slow EMA copy of the context encoder — encodes the actual next frame. It gets no gradient.
5. **Loss:** Mean squared error in embedding space, `0.5 * sum((predicted - target)^2)`, plus a variance loss. For each embedding dimension, if the standard deviation across the 5 context encodings is below 0.1, add a penalty proportional to the shortfall squared, weighted by 0.05. This pushes back against representation collapse.
6. **Backprop:** Backpropagate through the predictor to get a 160-dim context gradient, split it into 5 × 32-dim pieces, add the variance gradient, and backpropagate each piece through its cached encoder pass. Update the predictor and context encoder once per sliding window, online.
7. **EMA update:** After each window, update the target encoder: `target = 0.99 * target + 0.01 * context`, for weights and biases, unless the target encoder is frozen.

Training uses a sliding window: with 100 frames and 5 context frames, each trajectory gives 95 windows.

## Build and run

**Requirements:** GCC 16 with C++26 `<mdspan>` support (or a compatible compiler), and a terminal. This project uses `std::mdspan`, `std::jthread`, and `std::format`.

```bash
git clone https://github.com/cartercpp/JEPA.git
cd JEPA

g++-16 -std=c++26 -O3 -pthread \
  main.cpp jepa.cpp neural_network.cpp -o jepa

./jepa
```

Press **Enter** to stop training.

## Terminal visualization

Every 20 epochs, the program shows what the model is predicting, in two panels:

- **Blue `O`:** the real next position
- **Red `O`:** the predicted next position

The prediction is found by nearest neighbor: encode all 480 possible positions on the 16 × 30 display grid with the target encoder, and pick the position whose encoding is closest to the predicted encoding.

The display is 16 rows × 30 columns. Each panel redraws with ANSI escape sequences, with a 25 ms delay between frames.

## Source files

- `main.cpp` — ball simulation, training loop, and terminal visualization
- `jepa.hpp/.cpp` — context encoder, target encoder (EMA), predictor, sliding-window training, and variance loss
- `neural_network.hpp/.cpp` — feedforward network, manual backpropagation, SGD, He initialization, and `std::mdspan` matrix operations

## API overview

```cpp
jepa worldModel(
    {2, 128, 64, 32},              // encoder: input, hidden, embedding
    {32 * 5, 128, 64, 32},         // predictor: 5 encodings in, embedding out
    5,                             // contextFrames
    0.001f,                        // learningRate
    0.99f,                         // emaDecay
    0.05f,                         // varianceWeight
    0.1f,                          // minStdDev
    false                          // frozenTargetEncoder
);

float loss = worldModel.train(frames);
std::vector<float> encoding = worldModel.encode(frame, false);
std::vector<float> predicted = worldModel.predict(contextEncoding);
```

`train()` takes a vector of frames, where each frame is a 2-element vector. `encode()` uses the context encoder by default, or the target encoder when passed `true`. `predict()` takes a concatenated context encoding of size `contextFrames * embeddingSize`.

## Notes

This is an **educational, from-scratch implementation**, not a production world model. It learns from 2D position only, not pixels, so it does not test whether JEPA can learn from raw images. It uses online updates, one sliding window at a time, and has no quantitative evaluation yet beyond the loss and the live prediction visualization — no linear probe, no saved weights, and no automated tests.

The contract annotations in `jepa.hpp` and `neural_network.hpp` are currently commented out.

Background: JEPA as proposed by LeCun, and I-JEPA by Assran et al. — https://arxiv.org/abs/2301.08243

Author: [@cartercpp](https://github.com/cartercpp)
