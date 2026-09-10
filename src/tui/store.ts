/**
 * Everything the TUI reads off disk: the sweep table, the models it produced,
 * and the paper account.
 *
 * Kept apart from the renderer so a missing or stale file is a value the UI can
 * display rather than an exception thrown mid-frame. A trading tool that dies
 * because yesterday's sweep is not there is worse than one that says so.
 */

import { deserialize, type HmmParams } from "../hmm";
import { deserializeHsmm, type HsmmParams } from "../hsmm";
import { modelFileName, type SweepResult, type SweepRow } from "../sweep";
import * as hl from "../hyperliquid";
import type { CoinRow, ModelType } from "./model";
import type { RuntimeModel } from "./jobs";

export const DEFAULT_MODEL_DIR = "models";
export const DEFAULT_SWEEP_PATH = "models/sweep.json";
export const DEFAULT_PAPER_PATH = "models/paper.json";

export async function loadSweep(path = DEFAULT_SWEEP_PATH): Promise<SweepResult | null> {
  try {
    const f = Bun.file(path);
    if (!(await f.exists())) return null;
    return (await f.json()) as SweepResult;
  } catch {
    return null;
  }
}

/**
 * Merge the live market list with whatever the sweep decided.
 *
 * Ordering is by 24h volume, not by sweep rank. Ranking the list by excess ROI
 * would put the top of a 240-cell selection at the top of the screen every
 * time it opened, which is how a multiple-comparisons artefact becomes a
 * watchlist.
 */
export function rowsFrom(
  markets: hl.HlMarket[],
  sweep: SweepResult | null,
  defaults: { timeframe: string; modelType: ModelType },
): CoinRow[] {
  return markets.map((m) => {
    const best = sweep?.best?.[m.coin] as SweepRow | undefined;
    return {
      coin: m.coin,
      timeframe: best?.timeframe ?? defaults.timeframe,
      modelType: (best?.modelType ?? defaults.modelType) as ModelType,
      volume24h: m.dayNotionalVolume,
      markPrice: m.markPrice,
      excessRoi: best?.excessRoi ?? null,
      roi: best?.roi ?? null,
      buyHold: best?.buyHold ?? null,
      exposure: best?.exposure ?? null,
      trades: best?.trades ?? null,
      pValue: best?.pValue ?? null,
      note: best ? undefined : sweep ? "no qualifying row" : undefined,
    };
  });
}

/** Rehydrate a model written by `sweep --save` or `train --out`. */
export async function loadRuntimeModel(
  coin: string, timeframe: string, modelType: ModelType, dir = DEFAULT_MODEL_DIR,
): Promise<RuntimeModel | null> {
  const path = `${dir.replace(/\/$/, "")}/${modelFileName({ coin, timeframe, modelType } as SweepRow)}`;
  try {
    const f = Bun.file(path);
    if (!(await f.exists())) return null;
    const saved = await f.json();
    const params: HmmParams | HsmmParams = saved.modelType === "hsmm"
      ? deserializeHsmm(JSON.stringify(saved.params))
      : deserialize(JSON.stringify(saved.params));
    return {
      coin, timeframe, modelType: saved.modelType,
      params,
      scaler: {
        mean: new Float64Array(saved.scaler.mean),
        std: new Float64Array(saved.scaler.std),
      },
      names: saved.names,
      window: saved.window,
    };
  } catch {
    return null;
  }
}
