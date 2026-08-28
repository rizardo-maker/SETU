# Currency dataset layout

Collect your own images — the project document's whole argument for
rigor rests on this. Aim for 1,200-1,500+ images minimum, across:

- All denominations in circulation (both note series where they differ)
- 4 lighting conditions: daylight, indoor tube light, dim/evening, direct glare
- Folded, partial, crumpled, in-hand, flat on a table
- Both faces of every note, multiple backgrounds

Directory layout expected by `train_currency_classifier.py`
(an ImageFolder-compatible structure):

```
data/currency/
  train/
    10/    *.jpg
    20/    *.jpg
    50/    *.jpg
    100/   *.jpg
    200/   *.jpg
    500/   *.jpg
    2000/  *.jpg
  val/
    10/ ... (same class folders, held-out images)
    ...
```

Folder names become the class labels written to `models/currency_labels.json`
at training time — keep them as plain denomination strings ("10", "20", ...)
since `server/arbiter.py` speaks them directly ("<label> rupees").

Split roughly 80/20 train/val, and make sure the split is by *photo
session*, not just randomly by image — otherwise near-duplicate frames
of the same note leak between train and val and your validation
accuracy will lie to you.
