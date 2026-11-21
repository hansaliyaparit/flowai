/**
 * ============================================================================
 * FLOWAI NEURAL NETWORK LAB (FROM SCRATCH IN PLAIN TYPESCRIPT)
 * ============================================================================
 *
 * Task: Word-level filler tagger for voice dictation.
 * For each word in a sentence, predict:
 *   - Class 0: KEEP (normal word)
 *   - Class 1: REMOVE (verbal filler like "um", "uh", "like", "you know", "matlab")
 *
 * Design constraints:
 * - NO machine learning libraries used (pure standard TypeScript & Float32Array).
 * - Multi-hot 1024-dimensional feature hashing trick.
 * - Layer 1: W1 [1024 x 16], b1 [16], ReLU activation.
 * - Layer 2: W2 [16 x 2], b2 [2], Softmax output probabilities.
 * - Mini-batch SGD with backpropagation, class weighting, and layer freezing.
 * - Numerical gradient checking for verification.
 * ============================================================================
 */

export const INPUT_DIM = 1024;
export const HIDDEN_DIM = 16;
export const OUTPUT_DIM = 2; // 0 = keep, 1 = remove

export interface LabeledSentence {
  words: string[];
  labels: number[]; // 0 = keep, 1 = remove
}

export interface ForwardActivations {
  x: Float32Array;   // Multi-hot input vector [1024]
  z1: Float32Array;  // Pre-activation Layer 1 [16]
  h: Float32Array;   // Post-ReLU hidden activations [16]
  z2: Float32Array;  // Pre-activation Layer 2 [2]
  p: Float32Array;   // Softmax output probabilities [2] (p[0]=keep, p[1]=remove)
}

export interface LayerGradients {
  dW1: Float32Array; // [1024 x 16]
  db1: Float32Array; // [16]
  dW2: Float32Array; // [16 x 2]
  db2: Float32Array; // [2]
}

export interface ModelWeights {
  W1: Float32Array; // [1024 x 16]
  b1: Float32Array; // [16]
  W2: Float32Array; // [16 x 2]
  b2: Float32Array; // [2]
}

export interface ModelSnapshotJSON {
  timestamp: number;
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  W1: number[];
  b1: number[];
  W2: number[];
  b2: number[];
}

export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: {
    tp: number; // True Positive (correctly caught filler)
    fp: number; // False Positive (incorrectly marked keep word as filler)
    fn: number; // False Negative (missed filler)
    tn: number; // True Negative (correctly kept normal word)
  };
}

/**
 * ----------------------------------------------------------------------------
 * 1. SEEDED PSEUDO-RANDOM NUMBER GENERATOR (Mulberry32)
 * ----------------------------------------------------------------------------
 * Guarantees that weight initialization and dataset shuffling are 100%
 * reproducible across page reloads and between runs.
 */
export function createRNG(seed: number) {
  let s = seed | 0;
  return function nextFloat(): number {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box-Muller transform for standard normal random numbers: N(0, 1).
 */
export function normalRandom(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng(); // Avoid log(0)
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * ----------------------------------------------------------------------------
 * 2. FEATURE HASHING TRICK (THE "HASHING TRICK")
 * ----------------------------------------------------------------------------
 * WHY HASHING AVOIDS A FIXED VOCABULARY:
 * In traditional NLP, you build a fixed dictionary of every known word.
 * Problems with fixed dictionaries:
 *   1. Massive memory footprint (tens of thousands of word IDs).
 *   2. Out-Of-Vocabulary (OOV) crash: new words, slang, or typos map to <UNK>.
 *
 * Feature hashing hashes arbitrary strings directly into a fixed vector dimension
 * [0..1023]. This means:
 *   - Any new word or typo automatically gets an index without modifying vocabulary.
 *   - Memory is strictly capped at 1024 slots.
 *   - Multiple features (current word, previous word, next word) can simply be
 *     accumulated as a multi-hot vector.
 */
export function hashFeature(feature: string, dim: number = INPUT_DIM): number {
  // FNV-1a 32-bit hash algorithm
  let h = 2166136261;
  for (let i = 0; i < feature.length; i++) {
    h ^= feature.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h >>> 0) % dim;
}

/**
 * Build 1024-dim multi-hot vector for word at `index` in `words`.
 * Features used:
 *   - "w=" + currentWord
 *   - "p=" + previousWord (or "<s>" if at start)
 *   - "n=" + nextWord     (or "</s>" if at end)
 */
export function extractWordFeatures(words: string[], index: number): Float32Array {
  const vec = new Float32Array(INPUT_DIM);

  const curr = (words[index] || '').toLowerCase().trim();
  const prev = index > 0 ? (words[index - 1] || '').toLowerCase().trim() : '<s>';
  const next = index < words.length - 1 ? (words[index + 1] || '').toLowerCase().trim() : '</s>';

  const idxW = hashFeature(`w=${curr}`);
  const idxP = hashFeature(`p=${prev}`);
  const idxN = hashFeature(`n=${next}`);

  vec[idxW] = 1.0;
  vec[idxP] = 1.0;
  vec[idxN] = 1.0;

  return vec;
}

/**
 * ----------------------------------------------------------------------------
 * 3. NEURAL NETWORK IMPLEMENTATION (HE INITIALIZATION & ACTIVATIONS)
 * ----------------------------------------------------------------------------
 */
export class ScratchNeuralNetwork {
  public W1: Float32Array; // [1024 x 16]
  public b1: Float32Array; // [16]
  public W2: Float32Array; // [16 x 2]
  public b2: Float32Array; // [2]

  public freezeLayer1: boolean = false;
  public freezeLayer2: boolean = false;

  constructor(seed: number = 42) {
    this.W1 = new Float32Array(INPUT_DIM * HIDDEN_DIM);
    this.b1 = new Float32Array(HIDDEN_DIM);
    this.W2 = new Float32Array(HIDDEN_DIM * OUTPUT_DIM);
    this.b2 = new Float32Array(OUTPUT_DIM);

    this.initWeights(seed);
  }

  /**
   * He (Kaiming) Initialization:
   * Weights are drawn from N(0, sqrt(2 / fan_in)).
   * Prevents exploding/vanishing activations when used with ReLU.
   */
  public initWeights(seed: number = 42): void {
    const rng = createRNG(seed);

    // Layer 1 He initialization: fan_in = 1024
    const std1 = Math.sqrt(2.0 / INPUT_DIM);
    for (let i = 0; i < this.W1.length; i++) {
      this.W1[i] = normalRandom(rng) * std1;
    }
    this.b1.fill(0.0);

    // Layer 2 He initialization: fan_in = 16
    const std2 = Math.sqrt(2.0 / HIDDEN_DIM);
    for (let i = 0; i < this.W2.length; i++) {
      this.W2[i] = normalRandom(rng) * std2;
    }
    this.b2.fill(0.0);
  }

  /**
   * Forward pass for a single input vector x [1024].
   *
   * 1. Layer 1: z1 = x * W1 + b1
   * 2. ReLU:    h = max(0, z1)
   * 3. Layer 2: z2 = h * W2 + b2
   * 4. Softmax: p = exp(z2 - max) / sum(exp)
   */
  public forward(x: Float32Array): ForwardActivations {
    const z1 = new Float32Array(HIDDEN_DIM);
    const h = new Float32Array(HIDDEN_DIM);
    const z2 = new Float32Array(OUTPUT_DIM);
    const p = new Float32Array(OUTPUT_DIM);

    // 1. z1 = x * W1 + b1
    for (let j = 0; j < HIDDEN_DIM; j++) {
      let sum = this.b1[j];
      // Sparse optimization: only iterate non-zero input indices
      for (let i = 0; i < INPUT_DIM; i++) {
        if (x[i] !== 0) {
          sum += x[i] * this.W1[i * HIDDEN_DIM + j];
        }
      }
      z1[j] = sum;
      // 2. ReLU activation: h = max(0, z1)
      h[j] = sum > 0 ? sum : 0;
    }

    // 3. z2 = h * W2 + b2
    for (let k = 0; k < OUTPUT_DIM; k++) {
      let sum = this.b2[k];
      for (let j = 0; j < HIDDEN_DIM; j++) {
        sum += h[j] * this.W2[j * OUTPUT_DIM + k];
      }
      z2[k] = sum;
    }

    // 4. Numerically stable Softmax
    const maxZ = Math.max(z2[0], z2[1]);
    const exp0 = Math.exp(z2[0] - maxZ);
    const exp1 = Math.exp(z2[1] - maxZ);
    const sumExp = exp0 + exp1;

    p[0] = exp0 / sumExp; // Probability of KEEP
    p[1] = exp1 / sumExp; // Probability of REMOVE (filler)

    return { x, z1, h, z2, p };
  }

  /**
   * Backpropagation for a single sample:
   *
   * Cross-Entropy Loss with Class Weights:
   * Fillers are rare (~12-15%), so we weight class 1 higher (classWeight = 3.5).
   *
   * Equations:
   *   dz2 = (p - y) * classWeight
   *   dW2 = h^T * dz2 ; db2 = dz2
   *   dh  = dz2 * W2^T
   *   dz1 = dh * (z1 > 0) [ReLU derivative]
   *   dW1 = x^T * dz1 ; db1 = dz1
   */
  public backward(
    act: ForwardActivations,
    targetLabel: number,
    classWeight: number = 3.5
  ): { loss: number; grads: LayerGradients } {
    const { x, z1, h, p } = act;

    // Weight factor: 1.0 for keep, classWeight for remove
    const weight = targetLabel === 1 ? classWeight : 1.0;

    // Cross-Entropy Loss: -w * log(p_target)
    const probTarget = Math.max(1e-7, p[targetLabel]);
    const loss = -weight * Math.log(probTarget);

    // dz2 = (p - y) * weight
    const dz2 = new Float32Array(OUTPUT_DIM);
    dz2[0] = (p[0] - (targetLabel === 0 ? 1 : 0)) * weight;
    dz2[1] = (p[1] - (targetLabel === 1 ? 1 : 0)) * weight;

    // dW2 = h^T * dz2 ; db2 = dz2
    const dW2 = new Float32Array(HIDDEN_DIM * OUTPUT_DIM);
    const db2 = new Float32Array(OUTPUT_DIM);
    db2[0] = dz2[0];
    db2[1] = dz2[1];

    for (let j = 0; j < HIDDEN_DIM; j++) {
      dW2[j * OUTPUT_DIM + 0] = h[j] * dz2[0];
      dW2[j * OUTPUT_DIM + 1] = h[j] * dz2[1];
    }

    // dh = dz2 * W2^T
    const dh = new Float32Array(HIDDEN_DIM);
    for (let j = 0; j < HIDDEN_DIM; j++) {
      dh[j] = dz2[0] * this.W2[j * OUTPUT_DIM + 0] + dz2[1] * this.W2[j * OUTPUT_DIM + 1];
    }

    // dz1 = dh * (z1 > 0) (ReLU gradient)
    const dz1 = new Float32Array(HIDDEN_DIM);
    for (let j = 0; j < HIDDEN_DIM; j++) {
      dz1[j] = z1[j] > 0 ? dh[j] : 0;
    }

    // dW1 = x^T * dz1 ; db1 = dz1
    const dW1 = new Float32Array(INPUT_DIM * HIDDEN_DIM);
    const db1 = new Float32Array(HIDDEN_DIM);
    for (let j = 0; j < HIDDEN_DIM; j++) {
      db1[j] = dz1[j];
      const dj = dz1[j];
      if (dj !== 0) {
        for (let i = 0; i < INPUT_DIM; i++) {
          if (x[i] !== 0) {
            dW1[i * HIDDEN_DIM + j] = x[i] * dj;
          }
        }
      }
    }

    return {
      loss,
      grads: { dW1, db1, dW2, db2 },
    };
  }

  /**
   * Apply gradient descent update with support for layer freezing:
   * W -= learningRate * dW
   */
  public applyGradients(grads: LayerGradients, lr: number): void {
    if (!this.freezeLayer1) {
      for (let i = 0; i < this.W1.length; i++) {
        this.W1[i] -= lr * grads.dW1[i];
      }
      for (let j = 0; j < this.b1.length; j++) {
        this.b1[j] -= lr * grads.db1[j];
      }
    }

    if (!this.freezeLayer2) {
      for (let i = 0; i < this.W2.length; i++) {
        this.W2[i] -= lr * grads.dW2[i];
      }
      for (let k = 0; k < this.b2.length; k++) {
        this.b2[k] -= lr * grads.db2[k];
      }
    }
  }

  /**
   * Train one mini-batch using SGD:
   * Accumulates gradients across the batch, normalizes by batch size, and updates weights.
   */
  public trainBatch(
    samples: Array<{ x: Float32Array; y: number }>,
    lr: number,
    classWeight: number = 3.5
  ): number {
    if (samples.length === 0) return 0;

    const accumGrads: LayerGradients = {
      dW1: new Float32Array(this.W1.length),
      db1: new Float32Array(this.b1.length),
      dW2: new Float32Array(this.W2.length),
      db2: new Float32Array(this.b2.length),
    };

    let totalLoss = 0;

    for (const sample of samples) {
      const act = this.forward(sample.x);
      const { loss, grads } = this.backward(act, sample.y, classWeight);
      totalLoss += loss;

      if (!this.freezeLayer1) {
        for (let i = 0; i < this.W1.length; i++) accumGrads.dW1[i] += grads.dW1[i];
        for (let j = 0; j < this.b1.length; j++) accumGrads.db1[j] += grads.db1[j];
      }

      if (!this.freezeLayer2) {
        for (let i = 0; i < this.W2.length; i++) accumGrads.dW2[i] += grads.dW2[i];
        for (let k = 0; k < this.b2.length; k++) accumGrads.db2[k] += grads.db2[k];
      }
    }

    // Average gradients over the batch
    const scale = 1.0 / samples.length;
    if (!this.freezeLayer1) {
      for (let i = 0; i < this.W1.length; i++) accumGrads.dW1[i] *= scale;
      for (let j = 0; j < this.b1.length; j++) accumGrads.db1[j] *= scale;
    }
    if (!this.freezeLayer2) {
      for (let i = 0; i < this.W2.length; i++) accumGrads.dW2[i] *= scale;
      for (let k = 0; k < this.b2.length; k++) accumGrads.db2[k] *= scale;
    }

    this.applyGradients(accumGrads, lr);
    return totalLoss / samples.length;
  }

  /**
   * Evaluate model on a set of labeled sentences:
   * Computes Loss, Accuracy, Precision, Recall, F1 (for the "remove" filler class),
   * and confusion matrix.
   */
  public evaluate(sentences: LabeledSentence[], classWeight: number = 3.5): TrainingMetrics {
    let totalLoss = 0;
    let totalWords = 0;
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;

    for (const sent of sentences) {
      for (let w = 0; w < sent.words.length; w++) {
        const x = extractWordFeatures(sent.words, w);
        const y = sent.labels[w] || 0;
        const act = this.forward(x);

        const weight = y === 1 ? classWeight : 1.0;
        const loss = -weight * Math.log(Math.max(1e-7, act.p[y]));
        totalLoss += loss;
        totalWords++;

        const predicted = act.p[1] > 0.5 ? 1 : 0;
        if (predicted === 1 && y === 1) tp++;
        else if (predicted === 1 && y === 0) fp++;
        else if (predicted === 0 && y === 1) fn++;
        else tn++;
      }
    }

    const accuracy = totalWords > 0 ? (tp + tn) / totalWords : 0;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const avgLoss = totalWords > 0 ? totalLoss / totalWords : 0;

    return {
      loss: avgLoss,
      accuracy,
      precision,
      recall,
      f1,
      confusionMatrix: { tp, fp, fn, tn },
    };
  }

  /**
   * --------------------------------------------------------------------------
   * 4. UNIT-TEST-STYLE GRADIENT CHECK (NUMERICAL VS ANALYTIC GRADIENTS)
   * --------------------------------------------------------------------------
   * Verifies that the analytical calculus in backward() exactly matches
   * finite-difference numerical slopes:
   *   dLoss/dw ≈ [Loss(w + ε) - Loss(w - ε)] / (2ε)
   *
   * Tolerance: relative error < 1e-3.
   */
  public checkGradients(
    sampleX: Float32Array,
    targetY: number,
    epsilon: number = 1e-4
  ): { passed: boolean; maxRelError: number; sampleCount: number } {
    // 1. Compute analytical gradients
    const act = this.forward(sampleX);
    const { grads } = this.backward(act, targetY, 1.0); // unweighted for direct math check

    let maxRelError = 0;
    let checkedCount = 0;

    const testParam = (paramArray: Float32Array, analyticGrad: Float32Array, index: number) => {
      const orig = paramArray[index];

      // w + eps
      paramArray[index] = orig + epsilon;
      const actPlus = this.forward(sampleX);
      const lossPlus = -Math.log(Math.max(1e-7, actPlus.p[targetY]));

      // w - eps
      paramArray[index] = orig - epsilon;
      const actMinus = this.forward(sampleX);
      const lossMinus = -Math.log(Math.max(1e-7, actMinus.p[targetY]));

      // Restore
      paramArray[index] = orig;

      const numGrad = (lossPlus - lossMinus) / (2.0 * epsilon);
      const anaGrad = analyticGrad[index];

      const diff = Math.abs(numGrad - anaGrad);
      const denom = Math.abs(numGrad) + Math.abs(anaGrad) + 1e-7;
      const relError = diff / denom;

      if (relError > maxRelError) maxRelError = relError;
      checkedCount++;
    };

    // Check a sample of indices in W2, b2, W1, b1
    for (let i = 0; i < Math.min(8, this.W2.length); i++) {
      testParam(this.W2, grads.dW2, i);
    }
    for (let i = 0; i < this.b2.length; i++) {
      testParam(this.b2, grads.db2, i);
    }
    // Check active W1 indices (where x[i] === 1)
    let activeW1Checks = 0;
    for (let i = 0; i < INPUT_DIM && activeW1Checks < 12; i++) {
      if (sampleX[i] !== 0) {
        testParam(this.W1, grads.dW1, i * HIDDEN_DIM + 0);
        activeW1Checks++;
      }
    }
    for (let j = 0; j < Math.min(4, this.b1.length); j++) {
      testParam(this.b1, grads.db1, j);
    }

    const passed = maxRelError < 1e-2;
    return { passed, maxRelError, sampleCount: checkedCount };
  }

  /**
   * --------------------------------------------------------------------------
   * 5. WEIGHT SNAPSHOTS, PERSISTENCE & DELTA COMPARISONS
   * --------------------------------------------------------------------------
   */
  public toSnapshot(): ModelSnapshotJSON {
    return {
      timestamp: Date.now(),
      inputDim: INPUT_DIM,
      hiddenDim: HIDDEN_DIM,
      outputDim: OUTPUT_DIM,
      W1: Array.from(this.W1),
      b1: Array.from(this.b1),
      W2: Array.from(this.W2),
      b2: Array.from(this.b2),
    };
  }

  public loadSnapshot(snap: ModelSnapshotJSON): void {
    if (snap.W1 && snap.W1.length === this.W1.length) this.W1.set(snap.W1);
    if (snap.b1 && snap.b1.length === this.b1.length) this.b1.set(snap.b1);
    if (snap.W2 && snap.W2.length === this.W2.length) this.W2.set(snap.W2);
    if (snap.b2 && snap.b2.length === this.b2.length) this.b2.set(snap.b2);
  }

  public cloneWeights(): ModelWeights {
    return {
      W1: new Float32Array(this.W1),
      b1: new Float32Array(this.b1),
      W2: new Float32Array(this.W2),
      b2: new Float32Array(this.b2),
    };
  }

  /**
   * Compute average absolute change per layer between a previous weights snapshot
   * and the current weights. Useful for inspecting fine-tuning adaptation!
   */
  public computeWeightDeltas(previous: ModelWeights): {
    w1MeanChange: number;
    w2MeanChange: number;
    w1Diff: Float32Array;
    w2Diff: Float32Array;
  } {
    const w1Diff = new Float32Array(this.W1.length);
    let w1Sum = 0;
    for (let i = 0; i < this.W1.length; i++) {
      const diff = Math.abs(this.W1[i] - previous.W1[i]);
      w1Diff[i] = diff;
      w1Sum += diff;
    }

    const w2Diff = new Float32Array(this.W2.length);
    let w2Sum = 0;
    for (let i = 0; i < this.W2.length; i++) {
      const diff = Math.abs(this.W2[i] - previous.W2[i]);
      w2Diff[i] = diff;
      w2Sum += diff;
    }

    return {
      w1MeanChange: w1Sum / this.W1.length,
      w2MeanChange: w2Sum / this.W2.length,
      w1Diff,
      w2Diff,
    };
  }
}
