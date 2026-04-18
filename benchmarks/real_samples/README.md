# Real Benchmark Samples

This folder contains a tiny set of adapted samples from public agent and tool
use benchmarks. They use the repo's existing local JSON task protocol so the
smoke-test harness stays lightweight.

Sources:

- SWE-bench / SWE-bench Lite: real GitHub issue-resolution benchmark data,
  MIT licensed.
- Berkeley Function Calling Leaderboard (BFCL): function/tool-call benchmark
  data, Apache-2.0 licensed.

GAIA is not included because its dataset card asks users not to reshare the
gated validation or test data in a crawlable format.

These samples are not intended to reproduce the original benchmark metrics.
They are small real-data probes for the scaffold's model-input, context, and
output-handling behavior.
