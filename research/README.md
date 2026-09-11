# The test registry

Every hypothesis test run against this data gets a line in `tests.jsonl`, before
its result is known to anyone but the machine.

The reason is arithmetic. A p-value is the probability of seeing evidence this
strong *if nothing is there*, and it only means that if the test was the test
you were always going to run. Run twenty, keep the best, and 0.05 stops being a
one-in-twenty event and becomes a coin flip: at `n` independent tests the chance
that at least one lands under 0.05 is `1 - 0.95^n`, which passes 50% at n = 14
and 99% at n = 90. The repo's own sweep is the demonstration — 240 cells, best
p = 0.035, and *fewer* significant results than noise alone predicts.

So the discipline here is not "never look twice". It is: **look as often as you
like, and never lose count.** The registry is the count.

## Fields

| field | meaning |
| --- | --- |
| `id` | monotonic, assigned when the test is registered |
| `date` | when it ran |
| `hypothesis` | the claim in one sentence, written BEFORE the result |
| `test` | which procedure, and the null it draws from |
| `data` | universe, timeframe, window, cost |
| `config` | the model settings, fixed in advance |
| `pValue` | what came back |
| `verdict` | what it means once the running count is applied |
| `familySize` | how many tests this belongs to, cumulatively |

## Reading a p-value out of this file

The threshold is not 0.05. With `n` tests registered, a single result is only
interesting below roughly `0.05 / n` (Bonferroni — conservative but honest), and
Harvey, Liu & Zhu (2016) argue the bar in finance should be higher still because
the published record is itself a filtered sample. `src/spa.ts` computes the
correct joint answer for a searched-over family; this file is what tells it how
big the family is.
