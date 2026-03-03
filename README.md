# CTR Advanced Project

Проект для обучения и инференса модели предсказания клика (`is_click`) на `PyTorch` с CPU-only рантаймом.

## Что делает проект

- читает `train/test` CSV;
- строит признаки из `DateTime` (`hour`, `dayofweek`);
- обрабатывает пропуски;
- кодирует категориальные признаки через `OrdinalEncoder`;
- обучает MLP с embedding-слоями для категориальных фич;
- сохраняет артефакты обучения;
- запускает инференс на тесте и пишет `test_predictions.csv`.

## Стек

- Python 3.11+
- PyTorch
- pandas / numpy
- scikit-learn
- pytest / ruff / mypy

## Структура проекта

- `src/main.py` — train + inference pipeline
- `src/preprocessing.py` — препроцессинг и сохранение/загрузка артефактов
- `src/model.py` — dataset/model/early stopping
- `src/trainer.py` — training loop и checkpointing
- `src/inference.py` — сервис инференса по сохраненным артефактам
- `src/config.py` — конфигурация
- `tests/` — unit-тесты
- `data/` — входные CSV
- `artifacts/` — результаты запусков

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -e .[dev]
```

## Формат данных

Ожидается, что train содержит целевую колонку `is_click`, а test — без нее.

Ключевые поля, которые используются в коде:

- `DateTime`
- `gender`
- `product`
- `is_click` (только train)

`session_id` (если есть) удаляется перед обучением/инференсом.

## Запуск обучения и инференса

Базовый запуск:

```bash
python -m src.main
```

С явными путями:

```bash
python -m src.main \
  --train_path ./data/dataset_train.csv \
  --test_path ./data/dataset_test.csv \
  --artifacts_dir ./artifacts \
  --log_level INFO
```

После запуска артефакты сохраняются в:

`artifacts/<experiment_name>/<run_name>/`

Примеры файлов:

- `best.pt`, `last.pt`
- `scaler.joblib`, `cat_encoder.joblib`, `preprocessing_meta.json`
- `meta.json`, `history.json`, `train.log`
- `test_predictions.csv`

## Проверки качества

```bash
pytest -q
ruff check .
mypy
```

## Тесты

Покрыты базовые сценарии:

- корректность `Config`;
- `fit/transform/save/load` для `CTRPreprocessor`;
- базовые проверки `ClickDataset`, `ClickModel`, `EarlyStopping`.
