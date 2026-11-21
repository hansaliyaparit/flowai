import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Brain,
  Play,
  RotateCcw,
  Zap,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Activity,
  BarChart3,
  Sliders,
  Grid,
  FileText,
  BookOpen,
  ArrowRight,
  Download,
  Upload,
  Plus,
  RefreshCw,
} from 'lucide-react';
import {
  ScratchNeuralNetwork,
  extractWordFeatures,
  INPUT_DIM,
  HIDDEN_DIM,
  OUTPUT_DIM,
  type LabeledSentence,
  type TrainingMetrics,
  type ModelWeights,
  type ModelSnapshotJSON,
} from '../lab/nn';

const STORAGE_KEY_USER_SENTENCES = 'flowai_lab_user_sentences';
const STORAGE_KEY_WEIGHTS = 'flowai_lab_weights';

export const ModelLabPage: React.FC = () => {
  // Model instance held in a ref
  const modelRef = useRef<ScratchNeuralNetwork>(new ScratchNeuralNetwork(42));

  // Datasets
  const [baseSentences, setBaseSentences] = useState<LabeledSentence[]>([]);
  const [finetuneSentences, setFinetuneSentences] = useState<LabeledSentence[]>([]);
  const [userSentences, setUserSentences] = useState<LabeledSentence[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_USER_SENTENCES);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Hyperparameters
  const [trainSplit, setTrainSplit] = useState<number>(85); // 85% train, 15% val
  const [learningRate, setLearningRate] = useState<number>(0.05);
  const [epochs, setEpochs] = useState<number>(20);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [randomSeed, setRandomSeed] = useState<number>(42);

  // Training state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<Array<{ epoch: number; trainLoss: number; valLoss: number }>>([]);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);

  // Fine-tuning state
  const [freezeLayer1, setFreezeLayer1] = useState<boolean>(true);
  const [fineTuneLR, setFineTuneLR] = useState<number>(0.02);
  const [fineTuneEpochs, setFineTuneEpochs] = useState<number>(15);
  const [beforeSnapshot, setBeforeSnapshot] = useState<ModelWeights | null>(null);
  const [weightDeltas, setWeightDeltas] = useState<{
    w1MeanChange: number;
    w2MeanChange: number;
    w1Diff: Float32Array;
    w2Diff: Float32Array;
  } | null>(null);

  // Strategy comparison table
  const [strategyResults, setStrategyResults] = useState<Array<{
    strategy: string;
    loss: number;
    accuracy: number;
    recall: number;
    f1: number;
  }> | null>(null);

  // Prediction tester
  const [testInput, setTestInput] = useState<string>(
    'Um send the report basically by tomorrow you know.'
  );

  // Custom sentence builder
  const [newSentenceText, setNewSentenceText] = useState('');
  const [builderWords, setBuilderWords] = useState<string[]>([]);
  const [builderLabels, setBuilderLabels] = useState<number[]>([]);

  // Gradient check state
  const [gradCheckResult, setGradCheckResult] = useState<{
    passed: boolean;
    maxRelError: number;
    sampleCount: number;
  } | null>(null);

  // Canvas refs for weights heatmap
  const canvasW1Ref = useRef<HTMLCanvasElement | null>(null);
  const canvasW2Ref = useRef<HTMLCanvasElement | null>(null);
  const canvasDeltaW1Ref = useRef<HTMLCanvasElement | null>(null);

  // Active sub-tab
  const [activeTab, setActiveTab] = useState<'train' | 'finetune' | 'inspect' | 'predict' | 'data' | 'learn'>('train');

  // Load datasets on mount
  useEffect(() => {
    fetch('/data/base-dataset.jsonl')
      .then((r) => r.text())
      .then((text) => {
        const lines = text.split('\n').filter((l) => l.trim().length > 0);
        const parsed = lines.map((l) => JSON.parse(l) as LabeledSentence);
        setBaseSentences(parsed);
      })
      .catch((err) => console.error('Failed to load base dataset:', err));

    fetch('/data/finetune-dataset.jsonl')
      .then((r) => r.text())
      .then((text) => {
        const lines = text.split('\n').filter((l) => l.trim().length > 0);
        const parsed = lines.map((l) => JSON.parse(l) as LabeledSentence);
        setFinetuneSentences(parsed);
      })
      .catch((err) => console.error('Failed to load finetune dataset:', err));

    // Try loading saved weights
    try {
      const saved = localStorage.getItem(STORAGE_KEY_WEIGHTS);
      if (saved) {
        const snap = JSON.parse(saved) as ModelSnapshotJSON;
        modelRef.current.loadSnapshot(snap);
      }
    } catch {}
  }, []);

  // Save weights helper
  const persistWeights = useCallback(() => {
    try {
      const snap = modelRef.current.toSnapshot();
      localStorage.setItem(STORAGE_KEY_WEIGHTS, JSON.stringify(snap));
    } catch (e) {
      console.warn('Failed to save weights in localStorage', e);
    }
  }, []);

  // Split base dataset into train & validation
  const { trainBase, valBase } = useMemo(() => {
    if (baseSentences.length === 0) return { trainBase: [], valBase: [] };
    const splitIdx = Math.floor((baseSentences.length * trainSplit) / 100);
    return {
      trainBase: baseSentences.slice(0, splitIdx),
      valBase: baseSentences.slice(splitIdx),
    };
  }, [baseSentences, trainSplit]);

  // Combined fine-tune set (dataset + user additions)
  const combinedFinetuneSet = useMemo(() => {
    return [...finetuneSentences, ...userSentences];
  }, [finetuneSentences, userSentences]);

  // Render heatmap onto HTML Canvas
  const drawHeatmap = useCallback(
    (
      canvas: HTMLCanvasElement | null,
      weights: Float32Array,
      rows: number,
      cols: number,
      isDifference = false
    ) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      // Find max abs for color normalization
      let maxAbs = 0.01;
      for (let i = 0; i < weights.length; i++) {
        const absVal = Math.abs(weights[i]);
        if (absVal > maxAbs) maxAbs = absVal;
      }

      const cellW = width / cols;
      const cellH = height / rows;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const val = weights[r * cols + c];
          const norm = Math.min(1.0, Math.abs(val) / maxAbs);

          if (isDifference) {
            // Purple scale for delta change
            const intensity = Math.floor(norm * 255);
            ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.4)}, ${Math.floor(intensity * 0.9)})`;
          } else {
            if (val >= 0) {
              // Positive = Blue
              const b = Math.floor(norm * 240);
              ctx.fillStyle = `rgb(37, 99, ${b + 15})`;
            } else {
              // Negative = Amber / Red
              const red = Math.floor(norm * 240);
              ctx.fillStyle = `rgb(${red + 15}, 60, 60)`;
            }
          }
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
      }
    },
    []
  );

  // Redraw heatmaps whenever active tab is 'inspect' or weights update
  useEffect(() => {
    if (activeTab === 'inspect') {
      drawHeatmap(canvasW1Ref.current, modelRef.current.W1, 1024, 16);
      drawHeatmap(canvasW2Ref.current, modelRef.current.W2, 16, 2);
    }
  }, [activeTab, drawHeatmap, currentEpoch]);

  // Redraw delta heatmap
  useEffect(() => {
    if (weightDeltas && canvasDeltaW1Ref.current) {
      drawHeatmap(canvasDeltaW1Ref.current, weightDeltas.w1Diff, 1024, 16, true);
    }
  }, [weightDeltas, drawHeatmap]);

  // Reset Model Weights to initial He random distribution
  const handleResetWeights = () => {
    modelRef.current.initWeights(randomSeed);
    persistWeights();
    setLossHistory([]);
    setMetrics(null);
    setBeforeSnapshot(null);
    setWeightDeltas(null);
    setStrategyResults(null);
    setCurrentEpoch(0);
    if (activeTab === 'inspect') {
      drawHeatmap(canvasW1Ref.current, modelRef.current.W1, 1024, 16);
      drawHeatmap(canvasW2Ref.current, modelRef.current.W2, 16, 2);
    }
  };

  /**
   * TRAIN FROM SCRATCH (Base Dataset)
   * Non-blocking training loop: yields to browser event loop via setTimeout(0)
   * so UI animations and loss charts update at 60fps.
   */
  const handleTrainFromScratch = async () => {
    if (trainBase.length === 0 || isTraining) return;

    setIsTraining(true);
    modelRef.current.initWeights(randomSeed);
    modelRef.current.freezeLayer1 = false;
    modelRef.current.freezeLayer2 = false;

    // Flatten sentences into word samples: { x: Float32Array, y: number }
    const trainSamples: Array<{ x: Float32Array; y: number }> = [];
    for (const sent of trainBase) {
      for (let w = 0; w < sent.words.length; w++) {
        trainSamples.push({
          x: extractWordFeatures(sent.words, w),
          y: sent.labels[w] || 0,
        });
      }
    }

    const history: Array<{ epoch: number; trainLoss: number; valLoss: number }> = [];

    for (let ep = 1; ep <= epochs; ep++) {
      setCurrentEpoch(ep);

      // Shuffle samples for this epoch
      for (let i = trainSamples.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const temp = trainSamples[i];
        trainSamples[i] = trainSamples[j];
        trainSamples[j] = temp;
      }

      // Train in mini-batches
      let epochLossSum = 0;
      let batchCount = 0;
      for (let i = 0; i < trainSamples.length; i += batchSize) {
        const batch = trainSamples.slice(i, i + batchSize);
        const batchLoss = modelRef.current.trainBatch(batch, learningRate, 3.5);
        epochLossSum += batchLoss;
        batchCount++;
      }

      const trainAvgLoss = batchCount > 0 ? epochLossSum / batchCount : 0;
      const valMetrics = modelRef.current.evaluate(valBase, 3.5);

      history.push({
        epoch: ep,
        trainLoss: trainAvgLoss,
        valLoss: valMetrics.loss,
      });
      setLossHistory([...history]);
      setMetrics(valMetrics);

      // Yield thread so React updates UI
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    persistWeights();
    setIsTraining(false);
  };

  /**
   * FINE-TUNE (Hinglish Dataset + User Custom Sentences)
   */
  const handleFineTune = async () => {
    if (combinedFinetuneSet.length === 0 || isTraining) return;

    setIsTraining(true);
    // Take before-snapshot for weights difference inspector
    const snapshotBefore = modelRef.current.cloneWeights();
    setBeforeSnapshot(snapshotBefore);

    modelRef.current.freezeLayer1 = freezeLayer1;
    modelRef.current.freezeLayer2 = false;

    // Split fine-tune dataset: 80% train, 20% val
    const split = Math.floor(combinedFinetuneSet.length * 0.8);
    const ftTrain = combinedFinetuneSet.slice(0, split);
    const ftVal = combinedFinetuneSet.slice(split);

    const ftSamples: Array<{ x: Float32Array; y: number }> = [];
    for (const sent of ftTrain) {
      for (let w = 0; w < sent.words.length; w++) {
        ftSamples.push({
          x: extractWordFeatures(sent.words, w),
          y: sent.labels[w] || 0,
        });
      }
    }

    const history: Array<{ epoch: number; trainLoss: number; valLoss: number }> = [];

    for (let ep = 1; ep <= fineTuneEpochs; ep++) {
      setCurrentEpoch(ep);

      // Mini-batch updates
      let epochLossSum = 0;
      let batchCount = 0;
      for (let i = 0; i < ftSamples.length; i += 16) {
        const batch = ftSamples.slice(i, i + 16);
        const batchLoss = modelRef.current.trainBatch(batch, fineTuneLR, 3.5);
        epochLossSum += batchLoss;
        batchCount++;
      }

      const trainAvgLoss = batchCount > 0 ? epochLossSum / batchCount : 0;
      const valMetrics = modelRef.current.evaluate(ftVal, 3.5);

      history.push({
        epoch: ep,
        trainLoss: trainAvgLoss,
        valLoss: valMetrics.loss,
      });
      setLossHistory([...history]);
      setMetrics(valMetrics);

      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    // Compute delta change
    const deltas = modelRef.current.computeWeightDeltas(snapshotBefore);
    setWeightDeltas(deltas);

    persistWeights();
    setIsTraining(false);
  };

  /**
   * STRATEGY COMPARISON BENCHMARK (Feature 2 Requirement 7)
   * Compares 3 strategies on the fine-tune validation set:
   * (a) No fine-tuning (base model as-is)
   * (b) Fine-tune only Layer 2 (Layer 1 frozen)
   * (c) Fine-tune all layers at low LR
   */
  const handleRunStrategyComparison = async () => {
    if (combinedFinetuneSet.length === 0 || isTraining) return;

    setIsTraining(true);
    const split = Math.floor(combinedFinetuneSet.length * 0.8);
    const ftTrain = combinedFinetuneSet.slice(0, split);
    const ftVal = combinedFinetuneSet.slice(split);

    // Baseline snapshot
    const baselineSnap = modelRef.current.toSnapshot();

    // Strategy A: No fine-tuning
    const evalA = modelRef.current.evaluate(ftVal, 3.5);

    // Strategy B: Fine-tune only Layer 2
    modelRef.current.loadSnapshot(baselineSnap);
    modelRef.current.freezeLayer1 = true;
    modelRef.current.freezeLayer2 = false;
    const samplesB: Array<{ x: Float32Array; y: number }> = [];
    for (const sent of ftTrain) {
      for (let w = 0; w < sent.words.length; w++) {
        samplesB.push({ x: extractWordFeatures(sent.words, w), y: sent.labels[w] || 0 });
      }
    }
    for (let ep = 0; ep < 15; ep++) {
      for (let i = 0; i < samplesB.length; i += 16) {
        modelRef.current.trainBatch(samplesB.slice(i, i + 16), 0.05, 3.5);
      }
    }
    const evalB = modelRef.current.evaluate(ftVal, 3.5);

    // Strategy C: Fine-tune all layers at low LR
    modelRef.current.loadSnapshot(baselineSnap);
    modelRef.current.freezeLayer1 = false;
    modelRef.current.freezeLayer2 = false;
    for (let ep = 0; ep < 15; ep++) {
      for (let i = 0; i < samplesB.length; i += 16) {
        modelRef.current.trainBatch(samplesB.slice(i, i + 16), 0.008, 3.5);
      }
    }
    const evalC = modelRef.current.evaluate(ftVal, 3.5);

    // Restore baseline
    modelRef.current.loadSnapshot(baselineSnap);

    setStrategyResults([
      {
        strategy: '(a) No Fine-tuning (Base Model)',
        loss: evalA.loss,
        accuracy: evalA.accuracy,
        recall: evalA.recall,
        f1: evalA.f1,
      },
      {
        strategy: '(b) Fine-tune Only Layer 2 (Layer 1 Frozen)',
        loss: evalB.loss,
        accuracy: evalB.accuracy,
        recall: evalB.recall,
        f1: evalB.f1,
      },
      {
        strategy: '(c) Fine-tune All Layers (Low LR = 0.008)',
        loss: evalC.loss,
        accuracy: evalC.accuracy,
        recall: evalC.recall,
        f1: evalC.f1,
      },
    ]);

    setIsTraining(false);
  };

  // Run Gradient Check (Unit Test Style)
  const handleRunGradientCheck = () => {
    const testWords = ['um', 'please', 'like', 'review', 'this'];
    const x = extractWordFeatures(testWords, 0); // test on "um"
    const result = modelRef.current.checkGradients(x, 1, 1e-4);
    setGradCheckResult(result);
  };

  // Download weights as JSON
  const handleDownloadWeights = () => {
    const snap = modelRef.current.toSnapshot();
    const blob = new Blob([JSON.stringify(snap, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flowai-lab-weights-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Upload weights from JSON
  const handleUploadWeights = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const snap = JSON.parse(event.target?.result as string) as ModelSnapshotJSON;
        modelRef.current.loadSnapshot(snap);
        persistWeights();
        alert('Model weights loaded successfully!');
      } catch {
        alert('Invalid weights JSON file.');
      }
    };
    reader.readAsText(file);
  };

  // Custom Sentence Builder: prepare words
  const handlePrepareBuilderWords = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newSentenceText.trim();
    if (!trimmed) return;
    const words = trimmed.split(/\s+/);
    setBuilderWords(words);
    setBuilderLabels(words.map(() => 0)); // default keep (0)
  };

  const toggleBuilderLabel = (idx: number) => {
    setBuilderLabels((prev) => {
      const next = [...prev];
      next[idx] = next[idx] === 0 ? 1 : 0;
      return next;
    });
  };

  const handleSaveCustomSentence = () => {
    if (builderWords.length === 0) return;
    const newEntry: LabeledSentence = {
      words: builderWords,
      labels: builderLabels,
    };
    const next = [...userSentences, newEntry];
    setUserSentences(next);
    localStorage.setItem(STORAGE_KEY_USER_SENTENCES, JSON.stringify(next));
    setBuilderWords([]);
    setBuilderLabels([]);
    setNewSentenceText('');
  };

  // Prediction tester results
  const testPredictions = useMemo(() => {
    const words = testInput.trim().split(/\s+/).filter(Boolean);
    return words.map((word, idx) => {
      const x = extractWordFeatures(words, idx);
      const act = modelRef.current.forward(x);
      return {
        word,
        keepProb: act.p[0],
        removeProb: act.p[1],
        isFiller: act.p[1] > 0.5,
        hidden: Array.from(act.h),
      };
    });
  }, [testInput, currentEpoch, isTraining]);

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center text-white">
              <Brain className="w-4 h-4" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
              Model Lab: Neural Network from Scratch
            </h1>
          </div>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Build, inspect, backpropagate, and fine-tune a 2-layer neural network implemented in pure TypeScript.
          </p>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRunGradientCheck}
            className="inline-flex items-center gap-1.5 min-h-[36px] px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
            title="Verify calculus gradients numerically"
          >
            <Activity className="w-3.5 h-3.5 text-blue-500" />
            <span>Check Gradients</span>
          </button>

          <button
            type="button"
            onClick={handleDownloadWeights}
            className="inline-flex items-center gap-1.5 min-h-[36px] px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
            title="Download current model weights as JSON"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Weights JSON</span>
          </button>

          <label className="inline-flex items-center gap-1.5 min-h-[36px] px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer">
            <Upload className="w-3.5 h-3.5" />
            <span>Load JSON</span>
            <input type="file" onChange={handleUploadWeights} accept=".json" className="hidden" />
          </label>
        </div>
      </div>

      {/* Gradient Check Banner (if run) */}
      {gradCheckResult && (
        <div
          className={`p-3.5 rounded-xl border flex items-center justify-between gap-3 text-xs ${
            gradCheckResult.passed
              ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-800 text-emerald-900 dark:text-emerald-200'
              : 'bg-red-50 dark:bg-red-950/30 border-red-300 dark:border-red-800 text-red-900 dark:text-red-200'
          }`}
        >
          <div className="flex items-center gap-2">
            {gradCheckResult.passed ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            ) : (
              <XCircle className="w-4 h-4 text-red-600 shrink-0" />
            )}
            <span>
              <strong>Unit Test Gradient Check:</strong>{' '}
              {gradCheckResult.passed ? 'PASSED (Analytic gradients match numerical calculus)' : 'FAILED'} — Max relative error:{' '}
              <code className="font-mono">{gradCheckResult.maxRelError.toExponential(4)}</code> (across {gradCheckResult.sampleCount} checked weights).
            </span>
          </div>
          <button
            type="button"
            onClick={() => setGradCheckResult(null)}
            className="text-slate-400 hover:text-slate-600 cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Navigation Sub-Tabs */}
      <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 dark:bg-[#111827] rounded-xl border border-slate-200 dark:border-slate-800 text-xs font-medium">
        {[
          { id: 'train', label: '1. Training Panel', icon: Sliders },
          { id: 'predict', label: '2. Live Predictor', icon: Zap },
          { id: 'finetune', label: '3. Fine-Tune & Transfer', icon: RefreshCw },
          { id: 'inspect', label: '4. Weights Heatmap', icon: Grid },
          { id: 'data', label: '5. Dataset Hub', icon: FileText },
          { id: 'learn', label: '6. How It Works (Theory)', icon: BookOpen },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id as any)}
              className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-lg transition-colors cursor-pointer ${
                isActive
                  ? 'bg-white dark:bg-[#1F2937] text-blue-600 dark:text-blue-400 font-semibold shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* ==================================================================== */}
      {/* TAB 1: TRAINING PANEL & LIVE LOSS CHART                              */}
      {/* ==================================================================== */}
      {activeTab === 'train' && (
        <div className="space-y-6">
          {/* Controls & Metrics Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Hyperparameters Card */}
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Model Hyperparameters
                </h3>
                <span className="text-[11px] text-slate-400 font-mono">SGD Optimizer</span>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                    <span>Learning Rate (&eta;):</span>
                    <span className="font-mono font-bold text-blue-600">{learningRate}</span>
                  </div>
                  <input
                    type="range"
                    min="0.005"
                    max="0.2"
                    step="0.005"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    disabled={isTraining}
                    className="w-full accent-blue-600"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                    <span>Epochs:</span>
                    <span className="font-mono font-bold">{epochs}</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    step="5"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value, 10))}
                    disabled={isTraining}
                    className="w-full accent-blue-600"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                    <span>Mini-Batch Size:</span>
                    <span className="font-mono font-bold">{batchSize}</span>
                  </div>
                  <select
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value, 10))}
                    disabled={isTraining}
                    className="w-full p-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#0D1320]"
                  >
                    <option value={16}>16 samples / batch</option>
                    <option value={32}>32 samples / batch (Default)</option>
                    <option value={64}>64 samples / batch</option>
                  </select>
                </div>

                <div>
                  <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                    <span>Random Seed (Reproducibility):</span>
                    <span className="font-mono font-bold">{randomSeed}</span>
                  </div>
                  <input
                    type="number"
                    value={randomSeed}
                    onChange={(e) => setRandomSeed(parseInt(e.target.value, 10) || 42)}
                    disabled={isTraining}
                    className="w-full p-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#0D1320] font-mono"
                  />
                </div>

                <div className="pt-2 text-[11px] text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900/40 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800">
                  <strong>Architecture:</strong> W1 [1024 × 16] + ReLU &rarr; W2 [16 × 2] + Softmax. Total parameters: <strong>16,434</strong>.
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-2 flex flex-col gap-2">
                <button
                  type="button"
                  onClick={handleTrainFromScratch}
                  disabled={isTraining}
                  className={`w-full py-2.5 rounded-xl font-medium text-xs flex items-center justify-center gap-2 cursor-pointer transition-colors ${
                    isTraining
                      ? 'bg-blue-400 text-white cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white shadow-xs'
                  }`}
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{isTraining ? `Training (Epoch ${currentEpoch}/${epochs})…` : 'Train from Scratch'}</span>
                </button>

                <button
                  type="button"
                  onClick={handleResetWeights}
                  disabled={isTraining}
                  className="w-full py-2 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 font-medium text-xs flex items-center justify-center gap-2 cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Reset Weights (He Init)</span>
                </button>
              </div>
            </div>

            {/* Live Loss SVG Chart */}
            <div className="lg:col-span-2 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-blue-600" />
                    <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                      Live Training & Validation Loss Chart
                    </h3>
                  </div>
                  <div className="flex items-center gap-4 text-xs font-medium">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-blue-600" />
                      <span className="text-slate-600 dark:text-slate-300">Train Loss</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                      <span className="text-slate-600 dark:text-slate-300">Validation Loss</span>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Cross-entropy loss computed after each mini-batch epoch. Rendered in native SVG without external chart libraries.
                </p>
              </div>

              {/* Native SVG Chart Area */}
              <div className="my-4 h-48 w-full bg-slate-50 dark:bg-[#0A0F1D] rounded-xl border border-slate-100 dark:border-slate-800 p-2 relative">
                {lossHistory.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-xs text-slate-400">
                    Click &ldquo;Train from Scratch&rdquo; to watch the loss curve converge in real time.
                  </div>
                ) : (
                  <svg className="w-full h-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {/* Horizontal gridlines */}
                    <line x1="0" y1="25" x2="100" y2="25" stroke="#94a3b8" strokeDasharray="2" strokeWidth="0.5" opacity="0.3" />
                    <line x1="0" y1="50" x2="100" y2="50" stroke="#94a3b8" strokeDasharray="2" strokeWidth="0.5" opacity="0.3" />
                    <line x1="0" y1="75" x2="100" y2="75" stroke="#94a3b8" strokeDasharray="2" strokeWidth="0.5" opacity="0.3" />

                    {/* Compute points */}
                    {(() => {
                      const maxLoss = Math.max(...lossHistory.map((h) => Math.max(h.trainLoss, h.valLoss, 1.0)), 1.5);
                      const minLoss = 0;
                      const range = maxLoss - minLoss;

                      const trainPts = lossHistory.map((h, i) => {
                        const x = (i / Math.max(1, lossHistory.length - 1)) * 100;
                        const y = 100 - ((h.trainLoss - minLoss) / range) * 90 - 5;
                        return `${x.toFixed(1)},${y.toFixed(1)}`;
                      });

                      const valPts = lossHistory.map((h, i) => {
                        const x = (i / Math.max(1, lossHistory.length - 1)) * 100;
                        const y = 100 - ((h.valLoss - minLoss) / range) * 90 - 5;
                        return `${x.toFixed(1)},${y.toFixed(1)}`;
                      });

                      return (
                        <>
                          <polyline
                            fill="none"
                            stroke="#2563eb"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            points={trainPts.join(' ')}
                          />
                          <polyline
                            fill="none"
                            stroke="#10b981"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeDasharray="3 2"
                            points={valPts.join(' ')}
                          />
                        </>
                      );
                    })()}
                  </svg>
                )}
              </div>

              {/* Latest Epoch Status */}
              <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                <span>Epoch: {currentEpoch} / {epochs}</span>
                {lossHistory.length > 0 && (
                  <span>
                    Latest Train: {lossHistory[lossHistory.length - 1].trainLoss.toFixed(4)} | Val:{' '}
                    {lossHistory[lossHistory.length - 1].valLoss.toFixed(4)}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Metrics Panel & Confusion Matrix */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Evaluation Metrics (Remove Class)
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Why accuracy alone is misleading: In text dictation, filler words are only ~12-15% of all words. A dummy model predicting &ldquo;keep&rdquo; for every single word gets ~86% accuracy, but has 0% Recall and 0% F1 score!
                </p>
              </div>
            </div>

            {metrics ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-100 dark:border-slate-800 text-center">
                  <div className="text-[11px] font-medium text-slate-500">Overall Accuracy</div>
                  <div className="text-lg font-bold font-mono text-slate-900 dark:text-slate-100">
                    {(metrics.accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">(TP + TN) / Total</div>
                </div>

                <div className="p-3 rounded-xl bg-blue-50 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900/40 text-center">
                  <div className="text-[11px] font-medium text-blue-700 dark:text-blue-300">Precision</div>
                  <div className="text-lg font-bold font-mono text-blue-600 dark:text-blue-400">
                    {(metrics.precision * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-blue-500/80 mt-0.5">When flagged, was it filler?</div>
                </div>

                <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-100 dark:border-emerald-900/40 text-center">
                  <div className="text-[11px] font-medium text-emerald-700 dark:text-emerald-300">Recall</div>
                  <div className="text-lg font-bold font-mono text-emerald-600 dark:text-emerald-400">
                    {(metrics.recall * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-emerald-500/80 mt-0.5">What % of fillers did it catch?</div>
                </div>

                <div className="p-3 rounded-xl bg-purple-50 dark:bg-purple-950/30 border border-purple-100 dark:border-purple-900/40 text-center">
                  <div className="text-[11px] font-medium text-purple-700 dark:text-purple-300">F1 Score</div>
                  <div className="text-lg font-bold font-mono text-purple-600 dark:text-purple-400">
                    {metrics.f1.toFixed(3)}
                  </div>
                  <div className="text-[10px] text-purple-500/80 mt-0.5">Harmonic Mean (Target &gt; 0.85)</div>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-slate-400">
                Metrics will be calculated when you train or evaluate the model.
              </div>
            )}

            {/* Confusion Matrix */}
            {metrics && (
              <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800">
                <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 mb-2">
                  Confusion Matrix (Validation Set)
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800">
                    <div className="text-[10px] text-emerald-600">True Positives (TP)</div>
                    <div className="font-bold text-base text-emerald-700 dark:text-emerald-300">
                      {metrics.confusionMatrix.tp}
                    </div>
                    <div className="text-[10px] text-slate-400">Correctly removed filler</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800">
                    <div className="text-[10px] text-red-600">False Positives (FP)</div>
                    <div className="font-bold text-base text-red-700 dark:text-red-300">
                      {metrics.confusionMatrix.fp}
                    </div>
                    <div className="text-[10px] text-slate-400">Wrongly removed normal word</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800">
                    <div className="text-[10px] text-amber-600">False Negatives (FN)</div>
                    <div className="font-bold text-base text-amber-700 dark:text-amber-300">
                      {metrics.confusionMatrix.fn}
                    </div>
                    <div className="text-[10px] text-slate-400">Missed filler word</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800">
                    <div className="text-[10px] text-blue-600">True Negatives (TN)</div>
                    <div className="font-bold text-base text-blue-700 dark:text-blue-300">
                      {metrics.confusionMatrix.tn}
                    </div>
                    <div className="text-[10px] text-slate-400">Correctly kept normal word</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 2: LIVE PREDICTOR & HIDDEN ACTIVATIONS                           */}
      {/* ==================================================================== */}
      {activeTab === 'predict' && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-5">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
              Interactive Word-Level Prediction Tester
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Type or speak any sentence. The neural network featurizes each word in context, runs the forward pass, and outputs keep vs remove probabilities and 16-dimensional hidden activations.
            </p>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
              Sentence to Test:
            </label>
            <input
              type="text"
              value={testInput}
              onChange={(e) => setTestInput(e.target.value)}
              className="w-full p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0D1320] text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600/30"
              placeholder="Type a sentence with fillers..."
            />
          </div>

          {/* Prediction chips */}
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
              Token Probabilities &amp; Hidden Layer h &isin; &#8477;<sup>16</sup>
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {testPredictions.map((pred, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-xl border transition-all ${
                    pred.isFiller
                      ? 'border-red-300 dark:border-red-900/60 bg-red-50/40 dark:bg-red-950/20'
                      : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="font-mono font-bold text-sm text-slate-900 dark:text-slate-100">
                      &ldquo;{pred.word}&rdquo;
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        pred.isFiller
                          ? 'bg-red-600 text-white'
                          : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                      }`}
                    >
                      {pred.isFiller ? 'REMOVE (Filler)' : 'KEEP'}
                    </span>
                  </div>

                  {/* Horizontal Bar */}
                  <div className="space-y-1 text-[11px] font-mono">
                    <div className="flex justify-between text-slate-500">
                      <span>Keep: {(pred.keepProb * 100).toFixed(0)}%</span>
                      <span>Remove: {(pred.removeProb * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden flex">
                      <div className="h-full bg-emerald-500" style={{ width: `${pred.keepProb * 100}%` }} />
                      <div className="h-full bg-red-500" style={{ width: `${pred.removeProb * 100}%` }} />
                    </div>
                  </div>

                  {/* Mini Hidden Layer Activations (16 bars) */}
                  <div className="mt-3 pt-2 border-t border-slate-100 dark:border-slate-800">
                    <div className="text-[10px] text-slate-400 mb-1">ReLU Activations (h0 &hellip; h15):</div>
                    <div className="flex items-end gap-0.5 h-6">
                      {pred.hidden.map((val, idx) => {
                        const hHeight = Math.min(100, Math.max(8, val * 20));
                        return (
                          <div
                            key={idx}
                            title={`h[${idx}] = ${val.toFixed(3)}`}
                            className={`flex-1 rounded-xs transition-all ${
                              val > 0 ? 'bg-blue-600 dark:bg-blue-400' : 'bg-slate-300 dark:bg-slate-800'
                            }`}
                            style={{ height: `${hHeight}%` }}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 3: FINE-TUNE & TRANSFER LEARNING (HINGLISH ADAPTATION)           */}
      {/* ==================================================================== */}
      {activeTab === 'finetune' && (
        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Fine-Tuning on New Hinglish Fillers
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  The base model was trained on English fillers (um, uh, like, basically). Here we adapt the network to Hinglish fillers (matlab, yaar, haan, toh, actually, na) using Transfer Learning.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/40 space-y-2">
                <label className="flex items-center gap-2 font-medium text-slate-900 dark:text-slate-100 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={freezeLayer1}
                    onChange={(e) => setFreezeLayer1(e.target.checked)}
                    disabled={isTraining}
                    className="w-4 h-4 rounded text-blue-600"
                  />
                  <span>Freeze Layer 1 ($W_1, b_1$)</span>
                </label>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                  When frozen, the 1024-dimensional feature extraction weights are locked. Only the 16x2 classification head is updated. Prevents catastrophic forgetting!
                </p>
              </div>

              <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/40 space-y-2">
                <div className="flex justify-between font-medium">
                  <span>Fine-Tune LR:</span>
                  <span className="font-mono text-purple-600 font-bold">{fineTuneLR}</span>
                </div>
                <input
                  type="range"
                  min="0.005"
                  max="0.08"
                  step="0.005"
                  value={fineTuneLR}
                  onChange={(e) => setFineTuneLR(parseFloat(e.target.value))}
                  disabled={isTraining}
                  className="w-full accent-purple-600"
                />
                <p className="text-[11px] text-slate-500">Usually 2x–5x lower than base training rate.</p>
              </div>

              <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/40 space-y-2">
                <div className="flex justify-between font-medium">
                  <span>Fine-Tune Epochs:</span>
                  <span className="font-mono font-bold">{fineTuneEpochs}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="30"
                  step="5"
                  value={fineTuneEpochs}
                  onChange={(e) => setFineTuneEpochs(parseInt(e.target.value, 10))}
                  disabled={isTraining}
                  className="w-full accent-purple-600"
                />
                <p className="text-[11px] text-slate-500">Dataset size: {combinedFinetuneSet.length} sentences.</p>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 pt-2">
              <button
                type="button"
                onClick={handleFineTune}
                disabled={isTraining}
                className={`py-2 px-4 rounded-xl font-medium text-xs flex items-center gap-2 cursor-pointer transition-colors ${
                  isTraining ? 'bg-purple-400 text-white' : 'bg-purple-600 hover:bg-purple-700 text-white shadow-xs'
                }`}
              >
                <Zap className="w-3.5 h-3.5" />
                <span>{isTraining ? 'Fine-tuning…' : 'Run Fine-Tuning'}</span>
              </button>

              <button
                type="button"
                onClick={handleRunStrategyComparison}
                disabled={isTraining}
                className="py-2 px-4 rounded-xl border border-purple-200 dark:border-purple-800 text-purple-700 dark:text-purple-300 hover:bg-purple-50 dark:hover:bg-purple-950/40 font-medium text-xs flex items-center gap-2 cursor-pointer"
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Compare 3 Fine-Tuning Strategies</span>
              </button>
            </div>
          </div>

          {/* Strategy Comparison Results Table (Feature 2 Requirement 7) */}
          {strategyResults && (
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-3">
              <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                Strategy Comparison on Fine-Tune Validation Set
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 dark:bg-slate-800/50 text-slate-500 font-semibold border-b border-slate-200 dark:border-slate-700">
                    <tr>
                      <th className="p-3">Strategy</th>
                      <th className="p-3">Validation Loss</th>
                      <th className="p-3">Accuracy</th>
                      <th className="p-3">Recall (Caught Fillers)</th>
                      <th className="p-3">F1 Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono">
                    {strategyResults.map((res, i) => (
                      <tr key={i} className={i === 1 ? 'bg-purple-50/40 dark:bg-purple-950/20' : ''}>
                        <td className="p-3 font-sans font-medium text-slate-900 dark:text-slate-100">{res.strategy}</td>
                        <td className="p-3">{res.loss.toFixed(4)}</td>
                        <td className="p-3">{(res.accuracy * 100).toFixed(1)}%</td>
                        <td className="p-3">{(res.recall * 100).toFixed(1)}%</td>
                        <td className="p-3 font-bold text-purple-600 dark:text-purple-400">{res.f1.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Before & After Weights Delta Inspector (Feature 2 Requirement 8) */}
          {weightDeltas && (
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Weight Adaptation Deltas: |ΔW| = |W_after - W_before|
                </h4>
                <div className="flex items-center gap-3 text-xs font-mono">
                  <span>Layer 1 Mean Change: <strong>{weightDeltas.w1MeanChange.toExponential(3)}</strong></span>
                  <span>Layer 2 Mean Change: <strong>{weightDeltas.w2MeanChange.toFixed(4)}</strong></span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-xs text-slate-500">
                  Difference Heatmap for W1 (1024 × 16). Notice that if Layer 1 was frozen, changes are 0 (black).
                </div>
                <div className="h-28 w-full border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-black">
                  <canvas ref={canvasDeltaW1Ref} width={512} height={112} className="w-full h-full object-fill" />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 4: WEIGHTS INSPECTOR HEATMAP                                    */}
      {/* ==================================================================== */}
      {activeTab === 'inspect' && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-6">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
              Interactive Neural Network Weights Heatmap
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Visualizing the raw floating-point weights stored in Float32Array matrices. Blue = Positive weights; Red = Negative weights.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Layer 1: W1 [1024 x 16] */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-slate-700 dark:text-slate-200">
                  Layer 1 Weights: W1 [1024 × 16]
                </span>
                <span className="font-mono text-slate-400">16,384 params + 16 biases</span>
              </div>
              <div className="h-64 w-full border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden bg-slate-900">
                <canvas ref={canvasW1Ref} width={512} height={256} className="w-full h-full object-fill" />
              </div>
              <p className="text-[11px] text-slate-400">
                Each row corresponds to one of the 1024 hashed word/context feature buckets. Each column feeds one of the 16 hidden neurons.
              </p>
            </div>

            {/* Layer 2: W2 [16 x 2] */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-slate-700 dark:text-slate-200">
                  Layer 2 Weights: W2 [16 × 2]
                </span>
                <span className="font-mono text-slate-400">32 params + 2 biases</span>
              </div>
              <div className="h-64 w-full border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden bg-slate-900">
                <canvas ref={canvasW2Ref} width={256} height={256} className="w-full h-full object-fill" />
              </div>
              <p className="text-[11px] text-slate-400">
                Column 0: Logit weights for KEEP. Column 1: Logit weights for REMOVE (Filler).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 5: DATASET VIEWER & CUSTOM SENTENCE LABELER                     */}
      {/* ==================================================================== */}
      {activeTab === 'data' && (
        <div className="space-y-6">
          {/* Custom Sentence Annotation Tool */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-4">
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
              Add Your Own Custom Labeled Sentences (Active Learning)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Type any sentence. Words become interactive chips. Click any word to toggle its label between KEEP (green) and REMOVE (red), then add it to your fine-tune dataset!
            </p>

            <form onSubmit={handlePrepareBuilderWords} className="flex gap-2">
              <input
                type="text"
                value={newSentenceText}
                onChange={(e) => setNewSentenceText(e.target.value)}
                placeholder="Type a sentence with your own slang or fillers, e.g. 'yaar matlab this code is clean na'..."
                className="flex-1 p-2.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-[#0D1320] text-xs text-slate-900 dark:text-slate-100"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-medium cursor-pointer"
              >
                Parse Words
              </button>
            </form>

            {builderWords.length > 0 && (
              <div className="p-4 rounded-xl border border-blue-200 dark:border-blue-900/60 bg-blue-50/30 dark:bg-blue-950/20 space-y-3">
                <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Click words to toggle KEEP / REMOVE:
                </div>
                <div className="flex flex-wrap gap-2">
                  {builderWords.map((word, i) => {
                    const isFiller = builderLabels[i] === 1;
                    return (
                      <button
                        key={i}
                        type="button"
                        onClick={() => toggleBuilderLabel(i)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-colors cursor-pointer border ${
                          isFiller
                            ? 'bg-red-600 text-white border-red-700 shadow-xs'
                            : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300 border-emerald-300 dark:border-emerald-800'
                        }`}
                      >
                        {word} ({isFiller ? 'REMOVE' : 'KEEP'})
                      </button>
                    );
                  })}
                </div>
                <button
                  type="button"
                  onClick={handleSaveCustomSentence}
                  className="inline-flex items-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-medium cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Save to Fine-Tune Dataset</span>
                </button>
              </div>
            )}
          </div>

          {/* Dataset Table with Train/Val Split Slider */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Base Dataset ({baseSentences.length} Sentences)
                </h3>
                <p className="text-xs text-slate-500">
                  Loaded from <code>/data/base-dataset.jsonl</code> with random fillers inserted into clean business templates.
                </p>
              </div>

              <div className="flex items-center gap-3 text-xs">
                <span>Train / Val Split:</span>
                <input
                  type="range"
                  min="60"
                  max="90"
                  value={trainSplit}
                  onChange={(e) => setTrainSplit(parseInt(e.target.value, 10))}
                  className="w-28 accent-blue-600"
                />
                <span className="font-mono font-bold text-blue-600">
                  {trainSplit}% Train ({trainBase.length}) / {100 - trainSplit}% Val ({valBase.length})
                </span>
              </div>
            </div>

            <div className="max-h-96 overflow-y-auto divide-y divide-slate-100 dark:divide-slate-800">
              {baseSentences.slice(0, 50).map((sent, idx) => (
                <div key={idx} className="py-2.5 flex items-start gap-3 text-xs">
                  <span className="font-mono text-slate-400 w-8 shrink-0">#{idx + 1}</span>
                  <div className="flex flex-wrap gap-1.5">
                    {sent.words.map((w, wIdx) => {
                      const isFiller = sent.labels[wIdx] === 1;
                      return (
                        <span
                          key={wIdx}
                          className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                            isFiller
                              ? 'bg-red-100 text-red-700 dark:bg-red-950/60 dark:text-red-300 font-bold border border-red-200 dark:border-red-800'
                              : 'text-slate-700 dark:text-slate-300'
                          }`}
                        >
                          {w}
                        </span>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 6: HOW IT WORKS (THEORY & EDUCATION PANEL)                       */}
      {/* ==================================================================== */}
      {activeTab === 'learn' && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#111827] p-6 space-y-6">
          <div className="border-b border-slate-100 dark:border-slate-800 pb-3">
            <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
              Neural Network Foundations: Plain Language Explanations
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Deep understanding of every mathematical concept running in this Model Lab.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 space-y-1.5">
              <h4 className="font-bold text-blue-600 dark:text-blue-400">1. Weights and Biases ($W$ &amp; $b$)</h4>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Weights represent the strength and polarity of connections between neurons. A positive weight means a feature promotes a filler classification; a negative weight suppresses it. The bias ($b$) acts like a threshold, shifting the activation function regardless of input values.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 space-y-1.5">
              <h4 className="font-bold text-blue-600 dark:text-blue-400">2. ReLU Activation (max(0, z))</h4>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Without non-linear activations, stacking 100 neural network layers would mathematically collapse into a single linear matrix multiplication. Rectified Linear Unit (ReLU) sets negative values to 0, introducing non-linearity and allowing the network to learn complex boolean and combinatorial word relationships.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 space-y-1.5">
              <h4 className="font-bold text-blue-600 dark:text-blue-400">3. Softmax &amp; Cross-Entropy Loss</h4>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Softmax exponentiates raw output scores (logits) and divides by their sum so that outputs strictly sum to 1.0 (valid probability distribution). Cross-Entropy measures the distance between the predicted probability p and the true one-hot target y: Loss = -sum(y * log(p)).
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 space-y-1.5">
              <h4 className="font-bold text-blue-600 dark:text-blue-400">4. Backpropagation (Calculus Chain Rule)</h4>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Backprop calculates the partial derivative of the error with respect to every single weight (dL/dW). We use the chain rule backwards from output to input: dz2 = p - y, dh = dz2 * W2^T, and dz1 = dh * (z1 &gt; 0). Gradient descent then steps each weight downhill: W = W - learningRate * dW.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 space-y-1.5">
              <h4 className="font-bold text-blue-600 dark:text-blue-400">5. Layer Freezing &amp; Transfer Learning</h4>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                Layer 1 learns general grammatical patterns from 400 base sentences. When fine-tuning on a small Hinglish dataset (60 sentences), updating all 16,384 weights causes catastrophic forgetting. By freezing Layer 1, we preserve foundational word context and only adjust the final 34 classification weights.
              </p>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 space-y-1.5">
              <h4 className="font-bold text-blue-600 dark:text-blue-400">6. Overfitting vs Generalization</h4>
              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                If training loss keeps dropping but validation loss starts rising, the model is memorizing training sentences rather than learning general language rules. We combat this using train/val splits, mini-batch averaging, and moderate learning rates.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
