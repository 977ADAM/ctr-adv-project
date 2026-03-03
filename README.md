# CTR Advanced Project

Проект для обучения и инференса модели предсказания клика (`is_click`) на `PyTorch` с CPU-only рантаймом.

## Что делает проект

- читает `train/test` CSV;
- строит признаки из `DateTime` (`hour`, `dayofweek`);
- обрабатывает пропуски;
- типизирует признаки на числовые/категориальные;
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
- `src/api.py` — FastAPI mini-service (`/health`, `/predict`, `/model-info`)
- `src/config.py` — конфигурация
- `tests/` — unit-тесты
- `data/` — входные CSV
- `artifacts/` — результаты запусков
- `Dockerfile`, `docker-compose.yml` — контейнерный запуск демо

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -e .[dev]
pip install -e .[api]
```

## Формат данных

Ожидается, что train содержит целевую колонку `is_click`, а test — без нее.

Ключевые поля, которые используются в коде:

- `DateTime`
- `user_id`
- `gender`
- `product`
- `campaign_id`
- `webpage_id`
- `user_group_id`
- `product_category_1`
- `product_category_2`
- `is_click` (только train)

`session_id` (если есть) удаляется перед обучением/инференсом.

Для train/val split обязательно нужны `DateTime`, `user_id`, `is_click`.
Если этих колонок нет, `CTRPreprocessor.make_splits` завершится с явной ошибкой валидации схемы.

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

## Mini API (FastAPI)

Локальный запуск:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

По умолчанию сервис ищет последний валидный run в `artifacts/click_model/*`.
Можно явно указать путь к артефактам:

```bash
export MODEL_ARTIFACTS_DIR=./artifacts/click_model/<run_name>
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Эндпоинты:

- `GET /health` — статус сервиса и путь к артефактам (`degraded`, если модель не загрузилась)
- `GET /model-info` — информация о признаках/кардинальностях из `meta.json`
- `POST /predict` — инференс по батчу строк с ранней валидацией схемы

Схема `POST /predict` (`rows: list[PredictRow]`):

- обязательные поля:
  - `DateTime: str` (валидная datetime-строка, `null` запрещен)
  - `user_id: int | null`
  - `gender: str | null`
  - `product: str | null`
  - `campaign_id: int | null`
  - `webpage_id: int | null`
  - `user_group_id: float | int | null`
  - `product_category_1: float | int | null`
  - `product_category_2: float | int | null`
- дополнительные (`extra`) поля разрешены
- при ошибке схемы/типов API возвращает `422 Unprocessable Entity`

Пример `POST /predict`:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "DateTime": "2017-07-08 00:00",
        "user_id": 732573,
        "product": "J",
        "campaign_id": 404347,
        "webpage_id": 53587,
        "product_category_1": 1,
        "product_category_2": null,
        "user_group_id": 5,
        "gender": "Male",
        "age_level": 5,
        "user_depth": 3,
        "city_development_index": null,
        "var_1": 0
      }
    ]
  }'
```

## Docker Compose Demo

Запуск “из коробки”:

```bash
docker compose up --build
```

Сервис будет доступен на `http://localhost:8080`.

## Тесты

Покрыты базовые сценарии:

- корректность `Config`;
- `fit/transform/save/load` для `CTRPreprocessor`;
- валидация схемы и split без leakage (time + user_id);
- базовые проверки `ClickDataset`, `ClickModel`, `EarlyStopping`;
- `CTRInferenceService` для пустого/непустого input и CSV-выгрузки;
- `Trainer` для `resume_from` checkpoint.
