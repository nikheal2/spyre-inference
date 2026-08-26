# Temporary diagnostics — DELETE BEFORE MERGE

Scaffolding added while investigating **compile-mode output corruption** on the
`wrap_ministral` branch (eager is correct; `CompilationMode.STOCK_TORCH_COMPILE`
corrupts the last sequence of every batch, text and multimodal alike).

Every item is **off by default** — an unset env var or an unpassed CLI flag leaves
the production path byte-identical to before. Nothing here changes behaviour
unless explicitly switched on.

Find them all with:

```bash
grep -rn DIAGNOSTIC spyre_inference examples
```

## Removal checklist

### `spyre_inference/v1/attention/backends/spyre_attn.py`

- [ ] `_ONDEVICE_Q_ASSEMBLY` / `SPYRE_ATTN_ONDEVICE_Q` (~line 71) and its use in the
      `needs_query_cpu` condition. Default `"1"` = current behaviour.
      **Verdict: exonerated** — disabling it did not fix the corruption.

### `spyre_inference/v1/worker/spyre_model_runner.py`

- [ ] `_EMBED_ROUNDTRIP` / `SPYRE_EMBED_ROUNDTRIP` (~line 69) + its use in
      `embed_input_ids`. Default `0`.
- [ ] `_COMPILE_SCOPE` / `SPYRE_COMPILE_SCOPE` (~line 75), the early-return in
      `_compile_for_spyre`, and the whole `_compile_scoped` method. Default `"model"`.
      **Verdict: load-bearing evidence** — `mlp` clean vs `attn` corrupt localized
      the fault to `self_attn`.
- [ ] `_COMPILE_BACKEND` / `SPYRE_COMPILE_BACKEND` (~line 90). Default `"inductor"`.
      **Verdict: load-bearing evidence** — `eager`/`aot_eager` clean, `inductor`
      corrupt, which named Inductor codegen as the failing stage.
- [ ] `_PAD_BATCH_ROWS` / `SPYRE_PAD_BATCH_ROWS` + `_PADDABLE_INPUTS` (~line 96),
      the pad block at the top of `_SpyreModelWrapper.__call__`, the result crop
      after the model call, and the `_batch_pad_rows` / `_pad_rows` helpers.
      Default `0`. **Status: untested as of this writing.**
- [ ] The `"Spyre compile config: ..."` `logger.info` in `_compile_for_spyre`.

### `examples/offline_inference/torch_spyre_inference.py`

- [ ] `--mm-limit` flag + the conditional `limit_mm_per_prompt` kwarg it controls.
      **Verdict: exonerated** — text stayed coherent with it set.
- [ ] `--text-probe` flag + the `_text_probe` helper and its two call sites.
- [ ] `--image-url` flag + the `args.image_url` branch building `cases`.
- [ ] `--case-order` flag + the `all_cases` permutation block.
      **Verdict: load-bearing evidence** — proved the corruption follows batch
      position, not image content.

Keep `MULTIMODAL_CASES` and the `-n`-aware batching in `run_multimodal`; those are
genuine functionality, not diagnostics.

### `examples/offline_inference/`

- [ ] Delete `debug_gemv_m1.py` entirely.
      **Verdict: negative result** — all six op chains matched at every token
      count, which exonerated plain decoder math.

## Findings this scaffolding produced

1. Corruption is caused by `torch.compile`, not by OOT dispatch or `custom_ops`
   resolution (`SPYRE_COMPILE_SCOPE=none` is clean under an identical config).
2. It is localized to `self_attn`, not `mlp`.
3. It appears at the Inductor codegen stage; `eager` and `aot_eager` are clean.
4. It hits **the last sequence of the batch** at every batch size tested
   (1, 2, 3, 4) — originally misread as "batch size 1 is broken".
5. It is independent of input content: two slots given an identical image and
   question in one batch produced one clean and one corrupt result.
6. Multimodal is not separately broken. In compile mode the vision path also
   returns wrong image features for *all* sequences (fluent text describing a
   nonexistent image); eager is correct.
