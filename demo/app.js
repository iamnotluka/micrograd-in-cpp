const GRID_SIZE = 28;
const PIXEL_COUNT = GRID_SIZE * GRID_SIZE;
const MODEL_PATHS = ["../models/digits_mlp.params", "./digits_mlp.params"];

const canvas = document.querySelector("#digit-canvas");
const context = canvas.getContext("2d", { willReadFrequently: false });
const drawButton = document.querySelector("#draw-button");
const eraseButton = document.querySelector("#erase-button");
const resetButton = document.querySelector("#reset-button");
const brushSizeInput = document.querySelector("#brush-size");
const statusOutput = document.querySelector("#model-status");
const predictedDigitOutput = document.querySelector("#predicted-digit");
const confidenceList = document.querySelector("#confidence-list");

const pixels = new Float32Array(PIXEL_COUNT);
const confidenceRows = [];

let mode = "draw";
let brushRadius = Number.parseFloat(brushSizeInput.value);
let model = null;
let isPointerDown = false;
let lastPoint = null;
let predictionQueued = false;

class BrowserMLP {
  constructor(architecture, layers) {
    this.architecture = architecture;
    this.layers = layers;
  }

  static fromParams(text) {
    const values = text.trim().split(/\s+/).map(Number);
    let cursor = 0;

    const architectureLength = values[cursor++];
    const architecture = values.slice(cursor, cursor + architectureLength);
    cursor += architectureLength;

    const parameterCount = values[cursor++];
    const expectedParameterCount = BrowserMLP.countParameters(architecture);

    if (parameterCount !== expectedParameterCount) {
      throw new Error(`Expected ${expectedParameterCount} parameters, got ${parameterCount}`);
    }

    const layers = [];

    for (let layerIndex = 1; layerIndex < architecture.length; layerIndex++) {
      const inputCount = architecture[layerIndex - 1];
      const outputCount = architecture[layerIndex];
      const neurons = [];

      for (let neuronIndex = 0; neuronIndex < outputCount; neuronIndex++) {
        const weights = values.slice(cursor, cursor + inputCount);
        cursor += inputCount;
        const bias = values[cursor++];
        neurons.push({ weights, bias });
      }

      layers.push(neurons);
    }

    if (cursor !== values.length) {
      throw new Error("Model file has unexpected trailing values");
    }

    return new BrowserMLP(architecture, layers);
  }

  static countParameters(architecture) {
    let total = 0;

    for (let layerIndex = 1; layerIndex < architecture.length; layerIndex++) {
      total += architecture[layerIndex] * (architecture[layerIndex - 1] + 1);
    }

    return total;
  }

  forward(input) {
    if (input.length !== this.architecture[0]) {
      throw new Error(`Expected ${this.architecture[0]} inputs, got ${input.length}`);
    }

    let current = Array.from(input);

    for (const layer of this.layers) {
      current = layer.map((neuron) => {
        let activation = neuron.bias;

        for (let i = 0; i < neuron.weights.length; i++) {
          activation += neuron.weights[i] * current[i];
        }

        return Math.tanh(activation);
      });
    }

    return current;
  }
}

function initialiseConfidenceList() {
  const fragment = document.createDocumentFragment();

  for (let digit = 0; digit < 10; digit++) {
    const row = document.createElement("li");
    const label = document.createElement("span");
    const track = document.createElement("span");
    const fill = document.createElement("span");
    const value = document.createElement("span");

    row.className = "confidence-row";
    label.className = "digit-label";
    track.className = "confidence-track";
    fill.className = "confidence-fill";
    value.className = "confidence-value";

    label.textContent = String(digit);
    value.textContent = "0.0%";
    track.append(fill);
    row.append(label, track, value);
    fragment.append(row);
    confidenceRows.push({ row, fill, value });
  }

  confidenceList.append(fragment);
}

async function loadModel() {
  let lastError = null;

  for (const path of MODEL_PATHS) {
    try {
      const response = await fetch(path, { cache: "no-store" });

      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }

      const text = await response.text();
      model = BrowserMLP.fromParams(text);
      statusOutput.textContent = `${model.architecture.join(" -> ")} ready`;
      statusOutput.classList.add("is-ready");
      queuePrediction();
      return;
    } catch (error) {
      lastError = error;
    }
  }

  statusOutput.textContent = "model unavailable";
  statusOutput.classList.add("is-error");
  console.error("Could not load digit model", lastError);
}

function renderCanvas() {
  const image = context.createImageData(GRID_SIZE, GRID_SIZE);

  for (let i = 0; i < PIXEL_COUNT; i++) {
    const value = 255 - Math.round(pixels[i] * 255);
    const offset = i * 4;

    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }

  context.putImageData(image, 0, 0);
}

function setMode(nextMode) {
  mode = nextMode;
  const isDrawing = mode === "draw";

  drawButton.classList.toggle("is-active", isDrawing);
  eraseButton.classList.toggle("is-active", !isDrawing);
  drawButton.setAttribute("aria-pressed", String(isDrawing));
  eraseButton.setAttribute("aria-pressed", String(!isDrawing));
}

function getCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * GRID_SIZE;
  const y = ((event.clientY - rect.top) / rect.height) * GRID_SIZE;

  return {
    x: Math.max(0, Math.min(GRID_SIZE - 0.001, x)),
    y: Math.max(0, Math.min(GRID_SIZE - 0.001, y)),
  };
}

function applyBrush(point) {
  const radius = brushRadius;
  const minX = Math.max(0, Math.floor(point.x - radius - 1));
  const maxX = Math.min(GRID_SIZE - 1, Math.ceil(point.x + radius + 1));
  const minY = Math.max(0, Math.floor(point.y - radius - 1));
  const maxY = Math.min(GRID_SIZE - 1, Math.ceil(point.y + radius + 1));
  const strength = mode === "draw" ? 0.62 : -0.82;

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const centerX = x + 0.5;
      const centerY = y + 0.5;
      const distance = Math.hypot(centerX - point.x, centerY - point.y);

      if (distance > radius + 0.65) {
        continue;
      }

      const falloff = Math.max(0, 1 - distance / (radius + 0.65));
      const index = y * GRID_SIZE + x;
      pixels[index] = clamp(pixels[index] + strength * falloff, 0, 1);
    }
  }
}

function drawStroke(from, to) {
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const steps = Math.max(1, Math.ceil(distance * 2));

  for (let step = 0; step <= steps; step++) {
    const t = step / steps;
    applyBrush({
      x: from.x + (to.x - from.x) * t,
      y: from.y + (to.y - from.y) * t,
    });
  }

  renderCanvas();
  queuePrediction();
}

function queuePrediction() {
  if (predictionQueued) {
    return;
  }

  predictionQueued = true;
  requestAnimationFrame(() => {
    predictionQueued = false;
    updatePrediction();
  });
}

function updatePrediction() {
  const ink = pixels.reduce((total, value) => total + value, 0);

  if (!model || ink < 0.1) {
    predictedDigitOutput.textContent = "-";
    updateConfidenceBars(new Array(10).fill(0), -1);
    return;
  }

  const outputs = model.forward(pixels);
  const confidence = softmax(outputs, 4);
  const bestIndex = argmax(confidence);

  predictedDigitOutput.textContent = String(bestIndex);
  updateConfidenceBars(confidence, bestIndex);
}

function updateConfidenceBars(confidence, bestIndex) {
  confidence.forEach((value, digit) => {
    const percent = value * 100;
    const { row, fill, value: valueLabel } = confidenceRows[digit];

    row.classList.toggle("is-best", digit === bestIndex);
    fill.style.width = `${percent.toFixed(2)}%`;
    valueLabel.textContent = `${percent.toFixed(1)}%`;
  });
}

function softmax(values, scale = 1) {
  const scaled = values.map((value) => value * scale);
  const max = Math.max(...scaled);
  const exponents = scaled.map((value) => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);

  return exponents.map((value) => value / total);
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = values[0];

  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

drawButton.addEventListener("click", () => setMode("draw"));
eraseButton.addEventListener("click", () => setMode("erase"));

resetButton.addEventListener("click", () => {
  pixels.fill(0);
  renderCanvas();
  queuePrediction();
});

brushSizeInput.addEventListener("input", () => {
  brushRadius = Number.parseFloat(brushSizeInput.value);
});

canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  isPointerDown = true;
  lastPoint = getCanvasPoint(event);
  drawStroke(lastPoint, lastPoint);
});

canvas.addEventListener("pointermove", (event) => {
  if (!isPointerDown || !lastPoint) {
    return;
  }

  event.preventDefault();
  const point = getCanvasPoint(event);
  drawStroke(lastPoint, point);
  lastPoint = point;
});

canvas.addEventListener("pointerup", (event) => {
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }

  isPointerDown = false;
  lastPoint = null;
});

canvas.addEventListener("pointercancel", () => {
  isPointerDown = false;
  lastPoint = null;
});

initialiseConfidenceList();
renderCanvas();
loadModel();
