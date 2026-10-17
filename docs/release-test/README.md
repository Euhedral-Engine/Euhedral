# Release test registrations

One JSON file per run of the [release test](../RELEASE_TEST.md), committed before the run:

```json
{"date": "2026-10-20", "baseline": "q3",
 "candidates": {"q3": "<artifact sha256>", "q3-new": "<artifact sha256>"},
 "criteria": "q3-new is adopted if its overall KL is lower than q3's with the 95% interval below zero, and no domain's KL is more than 10% higher."}
```

`tools/t5/run.py STORE --registration docs/release-test/FILE.json NAME=ARTIFACT...` refuses a registration that is not
committed, and any artifact whose sha256 differs from the registered one.
