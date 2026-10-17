# R1Pro profile translations

`behavior_r1pro.md` is the runtime source of truth. Its human-readable Chinese
translation is `behavior_r1pro.zh-CN.md`; the Agent does not load the translation.

Whenever either file changes, review the other in the same change. When the
English source changes, update the Chinese wording and its embedded English
SHA-256. Keep action names, field names, coordinate frames, numeric limits,
and JSON examples identical in meaning. Do not update the hash alone to make
the check pass.

Run `python -m unittest scripts.Behavior_eval.checks.test_profile_translation`
from the repository root before finishing a change to either file. The full
`scripts/Behavior_eval/checks` suite includes this check as well.
