# tsprep

НИР по обучению представлений экономических временных рядов: TCN, LSTM,
Transformer и TCN+Attention. Исследуются сжатие окна и полезность эмбеддинга.
Реконструкция шумного входа сама по себе не доказывает шумоподавление.

## Установка

Python 3.10+, PowerShell, запуск из корня проекта:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
# Необязательные инструменты:
.venv/Scripts/python.exe -m pip install -e ".[dev,download,notebook]"
```

Runtime-зависимости: `reqs.txt`. Точные версии исследования:
`results/<study>/environment.txt`.

## Запуск

```powershell
.venv/Scripts/python.exe -m tsprep.cli --model hybrid_vae --epochs 40
.venv/Scripts/python.exe -m tsprep.cli --data csv --csv-path data/raw/aapl.csv --columns "Adj Close" --model tcn_ae
.venv/Scripts/python.exe -m tsprep.study --out results/my_new_study --epochs 30
.venv/Scripts/python.exe -m tsprep.cli --evaluate results/my_new_study/aapl/fold_0/tcn_ae_seed42_beta0
```

`--out` - новый каталог, не JSON-файл. Одиночный эксперимент не
перезаписывается. Исследование можно продолжить с теми же аргументами,
если протокол и исходники совпадают. Незавершённый отдельный артефакт
требует проверки причины сбоя.

Совместимый путь: `python scripts/run_experiment.py`.
После установки доступны `tsprep-experiment`, `tsprep-study`,
`tsprep-download`. Для VAE epochs должен превышать warmup (по умолчанию 5).
Используется GPU, если доступен; `--device cpu` выбирает CPU.
Ноутбуки 01–04 используют те же функции пакета.

Новые запуски используют формат v3 и строгий `--deterministic` по умолчанию.
TCN/Hybrid сохраняют линейную интерполяцию через фиксированную матрицу: это
избегает недетерминированного CUDA backward у `F.interpolate`. Строгий режим
проверен парными обучениями AE/VAE на CPU и CUDA; равенство между устройствами
или разными версиями библиотек не обещается. `--no-deterministic` явно отключает
строгую проверку операций.

В CLI `--data-seed` (по умолчанию 2026), `--model-seed` (`--seed`) и
`--loader-seed` разделены. Последний по умолчанию равен model seed.
В study `--data-seed` фиксирован между initialization repeats `--seeds`.

Продолжение v3 требует совпадения входов, меток, признаков, конфигурации,
вычислительных исходников и окружения. Study сохраняет неизменяемые snapshots,
архив кода, baseline receipts и связанные с весами metrics receipts. Во время
работы обновляется `progress.csv`; итоговые таблицы публикуются целиком перед
проверяемым `complete.json`. Завершённое исследование при resume проверяется
и возвращается без пересборки частичной таблицы. Одновременную запись блокирует
`.writer.lock`; после аварийного завершения процесса перед удалением оставшегося
lock нужно проверить записанные PID и host.

Исторические v2 study не продолжаются новым кодом. Их оценка требует явного
разрешения несовпадения исходников и сохраняет прежнюю интерполяцию:

```powershell
.venv/Scripts/python.exe -m tsprep.cli --evaluate results/rerun_2026-11-03/aapl/fold_0/tcn_ae_seed42_beta0 --allow-source-mismatch
.venv/Scripts/python.exe scripts/analyze_study.py results/rerun_2026-11-03 --out docs/my_historical_analysis --allow-source-mismatch
```

## Архитектуры

| Название | Энкодер | Декодер |
|---|---|---|
| tcn_ae / tcn_vae | TCN, mean pooling | интерполяция seed + TCN |
| lstm_ae / lstm_vae | LSTM, последнее состояние | авторегрессивный LSTM |
| transformer_ae / transformer_vae | Transformer, mean pooling | non-autoregressive Transformer |
| hybrid_ae / hybrid_vae | TCN + attention pooling | тот же TCN decoder |

Hybrid-AE и Transformer-VAE - абляции к шести исходным моделям.
Teacher forcing - опция TrainConfig только для LSTM, по умолчанию выключена.
В validation/test всегда используется свободная генерация.
Receptive field TCN: `1 + 2*(kernel_size-1)*(2**levels-1)`.
Attention не видит данные за границами окна.

## Данные

Для CSV признаки задаются явно через `--columns`. Числа должны быть
конечными, даты - уникальными и возрастающими. По умолчанию:
log → first difference → train-fitted Z-нормализация.
`--no-log-transform` выбирается осознанно для другой шкалы. Конвейер
предназначен для выбранных ценовых признаков, а не любых OHLCV/макропоказателей.

`csv_quality_report` проверяет пропуски, Inf, даты, постоянные признаки и
OHLC-согласованность. Подозрительные котировки автоматически не исправляются.
USD/RUB исключён из стандартного исследования: противоречивые котировки
и нулевой Volume требуют сверки источника. Raw CSV не изменяются.

Синтетика содержит random walk с GARCH-инновациями, сезонность, повторяющиеся
режимы, clean signal, наблюдаемый ряд и маску загрязнения. Clean signal
содержит случайные экономические инновации; из него исключены отдельно
добавленные ошибки наблюдения и аномальные импульсы.
Маска разностей учитывает обе точки вычитания.

## Протокол

Стандартный запуск: synthetic, AAPL (Adj Close), BTC (Close); окно 48,
stride 2, latent_dim 16, hidden_dim/d_model 32, TCN levels 3, batch 128;
seed 42, 43, 44. Тестовые периоды folds занимают примерно 70–85% и
85–100% истории и не пересекаются. Обучающие истории пересекаются.

Для каждого VAE на первом seed каждого fold сравниваются $\beta$=0.001 и 0.01
по validation reconstruction. Выбранный $\beta$ фиксируется для остальных seed.
Это ограниченный поиск, а не исчерпывающая оптимизация или сравнение при
равных вычислительных затратах. Всего 144 выбранных запуска и 24
дополнительных кандидата $\beta$, до 30 эпох на обучение.

Reconstruction усредняется по B*T*F, KL суммируется по latent_dim и
усредняется по B. $\beta$ задан в этих единицах. Checkpoint criterion -
неизменная deterministic reconstruction по μ, не Monte Carlo ELBO.
Для VAE выбор весов и отсчёт patience начинаются после warmup.
Early stopping ждёт 8 последовательных эпох без достаточного улучшения.

## Метрики и baseline

- MSE/MAE/R² в стандартизованной шкале и по признакам. Оконные метрики
  повторно учитывают пересекающиеся точки.
- Прогноз одного будущего преобразованного наблюдения на точном H=5.
  Это не цена и не накопленная доходность за H периодов. Ridge обучается
  на train; коэффициент регуляризации выбирается на validation.
- Нулевая реконструкция, PCA той же latent_dim, PCA+Ridge,
  raw-window Ridge, train mean/median и последнее преобразованное наблюдение.
- Silhouette с явным источником меток; один истинный режим не подменяется KMeans.
- USREC - ретроспективная классификация. Метки не заполняются из будущего.
  Для одного класса sensitivity/ROC-AUC ограничены или не определены.
- Anomaly score: MSE + 0.1*KL; порог - 95-й процентиль validation без
  подбора по test labels. Отдельно оцениваются концы окон и эпизоды
  на наблюдаемых концах окон.
- Эффективный ранг, active units, KL, дисперсии z. Отношение дисперсий
  зависит от масштаба и не измеряет информацию.
- На синтетике отдельно считается ошибка относительно clean signal.

Не считайте перекрывающиеся окна независимыми экспериментами.
Среднее/SD по seed и folds не заменяют интервалы с учётом зависимости:

```powershell
.venv/Scripts/python.exe scripts/analyze_study.py results/my_new_study
```

Скрипт сохраняет прогнозы, парные временные block-bootstrap интервалы
разницы с baseline и графики PNG/SVG. Интервалы условны на выбранных рядах
и протоколе; число временных блоков может быть небольшим.
Сравниваются mean, median, raw Ridge, PCA+Ridge и last observation. При менее
пяти временных блоках или block_len >= N границы интервала отсутствуют с явным
статусом `insufficient_temporal_blocks`; длина блока не обрезается до N.
Для короткого BTC test интервалы при блоках 24/48/96 не выдаются.
Все производные файлы помещаются в новый каталог анализа; `--out` позволяет
задать его явно. Clean MSE/MAE и ошибки на нормальных/загрязнённых точках
сохраняются отдельно. Перекрывающиеся точки в этих исторических метриках
по-прежнему учитываются повторно.

## Артефакты

Каждый запуск сохраняет `manifest.json`, `model.pt`, `input.npz`;
выбранный кандидат - также `metrics.json`. Manifest содержит полные
конфигурации, fitted normalizer, split bounds, признаки, историю,
best epoch/$\beta$, версии и контрольные суммы.
При загрузке preprocessing не переобучается, суммы проверяются.
Snapshot данных позволяет повторить оценку независимо от изменений CSV.
Загрузка также проверяет вычислительные source hashes. Изменения только
`evaluation/visualize.py` не меняют вычислительную идентичность. Новые artifacts
содержат `source.zip`, runtime/backend flags и `input_fingerprint`.

Исследование сохраняет protocol, data quality, baseline каждого fold,
подробную и агрегированную CSV-таблицы. Старые pickle/checkpoint-каталоги
автоматически не подхватываются: metadata недостаточно для воспроизведения.

## Проверки и структура

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py" -v
# При установленном dev extra:
.venv/Scripts/python.exe -m pytest -q
```

Проверяются прежние шесть функций, LSTM batch invariance, stopping/checkpoint,
horizon/stride, пороги, anomaly labels, preprocessing и повторная оценка.

`experiment.py` - общий конвейер; `study.py` - исследование;
`cli.py` - одиночный запуск. Подпакеты data_preprocessing, models,
mlcore и evaluation - данные, архитектуры, обучение и метрики.

## Ноутбуки и downstream

- `01_data_preparation.ipynb`: разметка режимов/аномалий, точные forecast-пары,
  paired observed/clean окна с одним train-fitted нормализатором.
- `02_model_zoo_smoke_test.ipynb`: формы и градиенты моделей, сравнение loss
  для observed и clean targets на одном noisy input.
- `03_training_and_comparison.ipynb`: стандартный study и отдельные таблицы
  задач по dataset/fold, включая baseline, precision/recall и доступность меток.
- `04_evaluation_and_results.ipynb`: forecast-графики, confusion matrix,
  PR-кривые, фиксированные validation anomaly thresholds, clean-ошибки на
  нормальных/загрязнённых endpoints и парные forecast-интервалы.
- `05_denoising_comparison.ipynb`: небольшой TCN-AE с двумя training targets,
  observed и clean; общий checkpoint criterion — validation endpoint clean MAE.
  Включены observed input, PCA, train mean и causal median baseline, а прогноз
  оценивается отдельно на точном H=7.

Ноутбук 05 — отладочный опыт с data seed 9001 и одним initialization seed.
Он не заменяет подтверждающий эксперимент на независимых траекториях.
Снижение MSE выбросов оценивается вместе с MAE и повреждением нормальных точек.

Парная цель доступна и для одиночного запуска на синтетике:

```powershell
.venv/Scripts/python.exe -m tsprep.cli --model tcn_ae --epochs 40 --data-seed 9001 --training-target clean --validation-target clean --selection-metric endpoint_mae
```

Clean target требуется явно; на реальном CSV без чистого эталона этот режим
завершится ошибкой. Старые сохранённые metrics можно читать при явном
`allow_source_mismatch=True`: их hashes и привязка к manifest проверяются,
но совпадение версии оценщика не требуется. Resume продолжает требовать
исходную конфигурацию и вычислительный код.
