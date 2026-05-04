# Results

No completed full-size reproduction results are bundled in this checkout.

Run:

```bash
python run_replication_parallel.py
```

The generated summaries will be saved to:

- `outputs/metrics/<task>_<base>to<target>_results.json`
- `outputs/metrics/all_results.json`

The reference TTNs reported in paper Table 1 are:

| Task | Pattern | Paper TTN |
|---|---|---:|
| A | TTT / RRT / TTR / RRR | 101 / 105 / 113 / 115 |
| B | TTTTT / TRTRT / RTRTR / RRRRR | 105 / 103 / 102 / 111 |
| C | TT / TR / RT / RR | 372 / 326 / 291 / 634 |
| D | TT / TR / RT / RR | 160 / 130 / 138 / 240 |
| E | TTTT / TRRT / RTTR / RRRR | 100 / 100 / 100 / 102 |
| F | TT / TR / RT / RR / BLE | 112 / 126 / 128 / 143 / 144 |
