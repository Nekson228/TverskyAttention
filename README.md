# TverskyAttention

ETU Diploma

## Установка

### Требования

- Python 3.13+
- Менеджер пакетов [uv](https://docs.astral.sh/uv/)

### Шаги

1. **Cклонировать репозиторий**

   ```bash
   git clone <repo-url>
   cd TverskyAttention
   ```

2. **Установить зависимости**

   ```bash
   uv sync
   ```

   Команда создаст виртуальное окружение и установит все зависимости из [`pyproject.toml`](pyproject.toml), включая PyTorch (CPU-сборка) из настроенного индекса.

3. **Запустить Jupyter-ноутбуки**

   ```bash
   uv run jupyter notebook
   ```

> **Примечание:** PyTorch устанавливается из CPU-only индекса (`https://download.pytorch.org/whl/cpu`). Для поддержки GPU обновите секцию `[tool.uv.sources]` в [`pyproject.toml`](pyproject.toml) и повторно выполните `uv sync`.

---

## Структура проекта

```
TverskyAttention/
├── data/                   # Локальное хранилище датасетов
│   ├── raw/                # Исходные данные
│   └── processed/          # Предобработанные и токенизированные данные
├── notebooks/              # Jupyter блокноты
├── src/                    # Основной исходный код проекта
│   ├── data_prep/          # Скрипты загрузки и подготовки данных
│   ├── architecture/       # Архитектура нейросетей
│   ├── train/              # Скрипты с циклами обучения
│   └── evaluation/         # Скрипты для оценки моделей
├── models/                 # Сохраненные чекпоинты и веса обученных моделей
├── thesis/                 # Материалы дипломной работы
│   ├── figures/            # Графики, диаграммы и визуализации внимания
│   └── report/             # Исходники текста ВКР
├── .gitignore
├── pyproject.toml
└── README.md               # Описание проекта и инструкции по запуску
```
