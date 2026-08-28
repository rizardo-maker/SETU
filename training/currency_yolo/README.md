# Currency YOLO training (Indian denominations)

Training scripts imported from the standalone `curr/` prototype. The runtime
detector now lives at `server/tier1/currency.py` — this directory is only for
(re)training the weights that ship as `models/currency_best.pt`.

## Layout
- `models/currency_best.pt` — trained YOLO weights, 10 classes
- `datasets/currency/combined/` — YOLO-format training set (1917 train + val)
- `datasets/currency/raw/Indian Currencies/` — original raw images by class
- `training/currency_yolo/train.py` — transfer-learning script (auto-picks MPS on Apple Silicon)

## Retrain
```
cd /Users/ravi/Documents/SETU
.venv/bin/python3 training/currency_yolo/train.py \
    --data datasets/currency/combined/data.yaml \
    --model models/yolo11n.pt \
    --save-dest models/currency_best.pt
```

## Classes (10 total)
`10_new`, `10_old`, `100_new`, `100_old`, `20`, `200`, `2000`, `50_new`, `50_old`, `500`
