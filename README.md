# tsprep

Реализация моделей предобработки временных рядов на основе нейросетевых
методов для экономических данных.

## Структура проекта

```
tsprep/
├── requirements.txt
├── src/tsprep/
│   ├── config.py                # DataConfig, ModelConfig, TrainConfig (все гиперпараметры)
│   ├── data_preprocessing/
│   │   ├── synthetic.py         # генератор синтетических экон. рядов (тренд+сезон+GARCH+режимы)
│   │   ├── preprocessing.py     # детрендирование + Z-нормализация (train-fit)
│   │   ├── datasets.py          # хронологический сплит, скользящее окно, DataLoader'ы
│   │   └── loaders.py           # загрузка реальных данных из CSV
│   ├── models/
│   │   ├── layers.py            # CausalConv1d/TemporalBlock (TCN), PositionalEncoding
│   │   ├── encoders.py          # TCNEncoder, LSTMEncoder, TransformerEncoder, HybridTCNAttentionEncoder
│   │   ├── decoders.py          # TCNDecoder, LSTMDecoder, TransformerDecoder
│   │   └── autoencoder.py       # Autoencoder/VAE-обертка + build_model(...) фабрика
│   ├── mlcore/
│   │   ├── losses.py            # recon loss, KL, KLAnnealer
│   │   └── trainer.py           # train_model(...): AdamW, grad clip, early stopping
│   └── evaluation/
│       ├── reconstruction.py    # MSE/MAE/R2, сбор эмбеддингов
│       ├── downstream.py        # linear probe (прогноз), silhouette, anomaly F1
│       ├── latent.py            # коэффициент коллапса $\mu$, PCA/t-SNE
│       └── visualize.py         # графики: кривые обучения, реконструкции, латентные проекции
├── scripts/
│   └── run_experiment.py        # CLI: полный прогон одной модели "данные -> метрики"
└── notebooks/
    ├── 01_data_preparation.ipynb
    ├── 02_model_zoo_smoke_test.ipynb
    ├── 03_training_and_comparison.ipynb
    └── 04_evaluation_and_results.ipynb
```

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Быстрый запуск одного эксперимента

```bash
python scripts/run_experiment.py --model hybrid_vae --epochs 40
python scripts/run_experiment.py --model tcn_ae --data csv --csv-path data/raw/my_series.csv
```

Результат (метрики реконструкции, forecast MAE, silhouette, anomaly F1,
коэффициент коллапса $\mu$) печатается в консоль и сохраняется в
`results/single_run.json`.

## Реализованные архитектуры (сравниваются в ноутбуке 03/04)

| model_name        | Энкодер              | Декодер      | VAE |
|--------------------|----------------------|--------------|-----|
| `tcn_ae`           | TCN                  | TCN          | нет |
| `tcn_vae`          | TCN                  | TCN          | да  |
| `lstm_ae`          | LSTM                 | LSTM (autoregressive) | нет |
| `lstm_vae`         | LSTM                 | LSTM (autoregressive) | да  |
| `transformer_ae`   | Transformer encoder  | Transformer (non-autoregressive) | нет |
| `hybrid_vae`        | **TCN + self-attention (предложенная модель)** | TCN | да  |

## Данные

- **Синтетика** (`data_preprocessing/synthetic.py`) - работает сразу, без интернета:
  стохастический тренд + многомасштабная сезонность + GARCH-подобная
  волатильность + структурные сдвиги режимов + внесенные точечные
  аномалии (для downstream-оценки).
- **Реальные данные** - положите CSV (дата + числовые колонки,
  отсортировано по возрастанию даты) и укажите путь через
  `--data csv --csv-path ...`. В этом окружении сетевой доступ к
  финансовым API (yfinance и т.п.) закрыт, поэтому выгрузку нужно
  сделать заранее, локально.

## Методология оценки

1. **Реконструкция**: MSE, MAE, $R^2$ на тестовом (по времени, после train/val) диапазоне.
2. **Полезность эмбеддингов** (downstream): линейный зонд z→y на горизонте H,
   silhouette эмбеддингов относительно скрытых режимов.
3. **Структура/устойчивость**: коэффициент информационного коллапса
   $\mu$ = tr(Cov(z))/tr(Cov(X)), F1 обнаружения внесенных аномалий по
   композитному скору `recon_error + λ·KL`.

## Примечание по интерпретации результатов

После дифференцирования (детрендирования) синтетический экономический
ряд близок к процессу с малой автокорреляцией (аналог гипотезы
эффективного рынка) - поэтому абсолютные значения $R^2$ на реконструкции
могут быть невысокими даже у хорошо обученной модели. Это ожидаемое,
методологически честное поведение, а не ошибка кода: сравнение
архитектур между собой (относительные MSE/$\mu$/F1) остается содержательным.