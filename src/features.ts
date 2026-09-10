/**
 * Turning candles into the feature vector the HMM sees.
 *
 * Three features, all computable from data available at the close of bar i —
 * nothing here peeks at bar i+1:
 *   0. log return            — direction. Kept as feature 0 because state
 *                              sorting and the trading signal both key off it.
 *   1. realized volatility   — meme coins switch between dead and unhinged;
 *                              this is usually the strongest regime separator.
 *   2. volume surge          — log(volume / rolling mean volume). A pump with
 *                              no volume behind it is noise.
 */

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FeatureConfig {
  /** Lookback for realized vol and the volume baseline. */
  window?: number;
  useVolatility?: boolean;
  useVolume?: boolean;
}

export interface FeatureSet {
  /** Flat T*D matrix of standardized features. */
  X: Float64Array;
  T: number;
  D: number;
  names: string[];
  /** index[i] = position in the original candle array that row i was built from. */
  index: Int32Array;
  /** Raw (unstandardized) log return for each row, for readable diagnostics. */
  rawReturn: Float64Array;
}

export interface Scaler {
  mean: Float64Array;
  std: Float64Array;
}

/** Build unstandardized features. Rows with an incomplete lookback are dropped. */
export function buildFeatures(candles: Candle[], config: FeatureConfig = {}): FeatureSet {
  const window = config.window ?? 20;
  const useVol = config.useVolatility ?? true;
  const useVolume = config.useVolume ?? true;

  const n = candles.length;
  if (n < window + 2) throw new Error(`need at least ${window + 2} candles, got ${n}`);

  const names = ["logReturn"];
  if (useVol) names.push("realizedVol");
  if (useVolume) names.push("volumeSurge");
  const D = names.length;

  // Log returns first; everything else is a function of these plus volume.
  const ret = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const prev = Math.max(candles[i - 1].close, 1e-18);
    const cur = Math.max(candles[i].close, 1e-18);
    ret[i] = Math.log(cur / prev);
  }

  const start = window; // first index with a full lookback
  const T = n - start;
  const X = new Float64Array(T * D);
  const index = new Int32Array(T);
  const rawReturn = new Float64Array(T);

  for (let i = start; i < n; i++) {
    const row = i - start;
    index[row] = i;
    rawReturn[row] = ret[i];

    let d = 0;
    X[row * D + d++] = ret[i];

    if (useVol) {
      // Realized vol over the trailing window, inclusive of the current bar.
      let mean = 0;
      for (let j = i - window + 1; j <= i; j++) mean += ret[j];
      mean /= window;
      let sq = 0;
      for (let j = i - window + 1; j <= i; j++) {
        const dv = ret[j] - mean;
        sq += dv * dv;
      }
      // log-vol rather than vol: the distribution is far closer to Gaussian,
      // which is exactly what the emission model assumes.
      X[row * D + d++] = Math.log(Math.sqrt(sq / window) + 1e-8);
    }

    if (useVolume) {
      let vsum = 0;
      for (let j = i - window + 1; j <= i; j++) vsum += Math.max(candles[j].volume, 0);
      const baseline = vsum / window;
      X[row * D + d++] = Math.log((Math.max(candles[i].volume, 0) + 1e-8) / (baseline + 1e-8));
    }
  }

  return { X, T, D, names, index, rawReturn };
}

/** Fit a standardizer on training rows only — never on the full series. */
export function fitScaler(X: Float64Array, T: number, D: number): Scaler {
  const mean = new Float64Array(D);
  const std = new Float64Array(D);
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) mean[d] += X[t * D + d];
  for (let d = 0; d < D; d++) mean[d] /= T;
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      const diff = X[t * D + d] - mean[d];
      std[d] += diff * diff;
    }
  }
  for (let d = 0; d < D; d++) std[d] = Math.max(Math.sqrt(std[d] / T), 1e-12);
  return { mean, std };
}

export function applyScaler(X: Float64Array, T: number, D: number, s: Scaler): Float64Array {
  const out = new Float64Array(T * D);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) out[t * D + d] = (X[t * D + d] - s.mean[d]) / s.std[d];
  }
  return out;
}

/** Undo standardization for feature d — used to read state means in real units. */
export function unscale(value: number, d: number, s: Scaler): number {
  return value * s.std[d] + s.mean[d];
}

/** Slice a flat T*D matrix, [from, to). */
export function sliceRows(X: Float64Array, D: number, from: number, to: number): Float64Array {
  return X.slice(from * D, to * D);
}
