/**
 * The wire between the UI thread and the compute thread.
 *
 * Fitting a 3-state HSMM over 3000 bars with 8 restarts takes seconds, and a
 * walk-forward does it once per block. Run that on the render thread and the
 * terminal freezes mid-keystroke — no spinner, no cancel, no repaint. So every
 * expensive call goes to a Worker and comes back as a message, and the UI keeps
 * drawing the whole time.
 *
 * Requests carry an id; responses echo it. Progress is a separate message kind
 * rather than a callback, because callbacks do not survive structured clone.
 */

import type { Candle } from "../features";
import type { BacktestView, ChartView, FitView, ModelType } from "./model";

/**
 * A fitted model in the shape the paper engine and `signalNow` consume — live
 * Float64Arrays, not the JSON that `sweep.SavedModel` writes to disk. Both
 * exist on purpose: one crosses a worker boundary, the other crosses a restart.
 */
export interface RuntimeModel {
  coin: string;
  timeframe: string;
  modelType: ModelType;
  params: unknown;
  scaler: { mean: Float64Array; std: Float64Array };
  names: string[];
  window: number;
}

export type JobRequest =
  | { id: number; kind: "candles"; coin: string; timeframe: string; bars: number }
  | {
      id: number; kind: "fit";
      coin: string; timeframe: string; bars: number;
      states: number; modelType: ModelType; maxDuration: number; seed: number; restarts: number;
    }
  | {
      id: number; kind: "backtest";
      coin: string; timeframe: string; bars: number;
      states: number; modelType: ModelType; maxDuration: number;
      costBps: number; trials: number; seed: number; restarts: number;
    }
  | {
      id: number; kind: "sweep";
      limit: number; timeframes: string[]; days: number; costBps: number; trials: number; coins?: string[];
    };

export interface CandlesResult { candles: Candle[] }
export interface FitResult { fit: FitView; chart: ChartView; model: RuntimeModel }
export interface BacktestResultMsg { backtest: BacktestView; chart: ChartView; model: RuntimeModel }
export interface SweepResultMsg { rows: unknown[]; best: Record<string, unknown> }

/** Omit that distributes over the union — a plain Omit<JobRequest, "id"> would
 *  collapse the four request shapes into their common fields. */
type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
export type JobRequestInput = DistributiveOmit<JobRequest, "id">;

export type JobResponse =
  | { id: number; kind: "progress"; done: number; total: number; label: string }
  | { id: number; kind: "ok"; result: unknown }
  | { id: number; kind: "error"; error: string };

/**
 * Client side. One worker, many in-flight jobs keyed by id.
 *
 * `cancel` does not kill work already running inside the worker — a synchronous
 * EM fit cannot be interrupted — it just stops the UI from caring about the
 * answer. That is the honest guarantee, and the UI reflects it: the spinner
 * clears, but a later result for a stale id is dropped rather than shown.
 */
export class JobClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, {
    resolve: (v: any) => void;
    reject: (e: Error) => void;
    onProgress?: (done: number, total: number, label: string) => void;
  }>();

  constructor(url = new URL("./worker.ts", import.meta.url).href) {
    this.worker = new Worker(url, { type: "module" });
    this.worker.addEventListener("message", (e: MessageEvent) => {
      const msg = e.data as JobResponse;
      const entry = this.pending.get(msg.id);
      if (!entry) return; // cancelled, or a late duplicate
      if (msg.kind === "progress") { entry.onProgress?.(msg.done, msg.total, msg.label); return; }
      this.pending.delete(msg.id);
      if (msg.kind === "ok") entry.resolve(msg.result);
      else entry.reject(new Error(msg.error));
    });
    this.worker.addEventListener("error", (e: any) => {
      const err = new Error(String(e?.message ?? "worker crashed"));
      for (const [, entry] of this.pending) entry.reject(err);
      this.pending.clear();
    });
  }

  run<T>(
    req: JobRequestInput,
    onProgress?: (done: number, total: number, label: string) => void,
  ): { id: number; promise: Promise<T> } {
    const id = this.nextId++;
    const promise = new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ ...req, id } as JobRequest);
    });
    return { id, promise };
  }

  /** Stop caring about a job. See the class comment for what this cannot do. */
  cancel(id: number) {
    this.pending.get(id)?.reject(new Error("cancelled"));
    this.pending.delete(id);
  }

  terminate() {
    for (const [, entry] of this.pending) entry.reject(new Error("terminated"));
    this.pending.clear();
    this.worker.terminate();
  }
}
