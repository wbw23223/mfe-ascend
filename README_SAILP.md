# SAI-LP patch for `micavro/mfe-ascend`

This overlay adds a SAIL/SAI-LP state-aware scheduler to MFE Ascend.

Apply from the root of your fork/clone:

```bash
python apply_sailp_changes.py
python -m py_compile \
  mfe/optimizers/sailp.py \
  mfe/optimizers/multi_request.py \
  mfe/parser.py \
  mfe/components/operator.py \
  mfe/components/model_config.py \
  mfe/components/query.py \
  mfe/workers/worker_v.py
```

Then run:

```bash
export MFE_SCHEDULER=sailp
export MFE_ENABLE_PREFIX_CACHING=1
python -m mfe.scripts.client --dataset gsm8k -n 5 --yaml adv_reason_3.yaml --test-worker --worker-delay 0.2 -v
```

See `docs/sailp.md` for YAML metadata, environment variables, and limitations.
