# TverskyAttention

ETU Diploma

## Установка

### Требования

- Python 3.13+
- Менеджер пакетов [uv](https://docs.astral.sh/uv/)

### Шаги

1. **Cклонировать репозиторий**

   ```bash
   git clone https://github.com/Nekson228/TverskyAttention.git
   cd TverskyAttention
   ```

2. **Установить зависимости**

   ```bash
   uv sync
   ```

   Команда создаст виртуальное окружение и установит все зависимости из [`pyproject.toml`](pyproject.toml).

3. **Запустить Jupyter-ноутбуки**

   ```bash
   uv run jupyter notebook
   ```

> **Примечание:** PyTorch устанавливается из [CPU-only индекса](https://download.pytorch.org/whl/cpu). Для поддержки GPU обновите секцию `[tool.uv.sources]` в [`pyproject.toml`](pyproject.toml) и повторно выполните `uv sync`.

---

## Структура проекта

```
TverskyAttention/
├── data/                   # Локальное хранилище датасетов
│   └── processed/          # Предобработанные и токенизированные данные
├── notebooks/              # Jupyter блокноты
├── src/                    # Основной исходный код проекта
│   ├── datasets/           # Скрипты загрузки и подготовки данных
│   └── architecture/       # Архитектура нейросетей
├── models/                 # Сохраненные веса обученных моделей
├── thesis/                 # Материалы дипломной работы
│   └── resources/          # Графики, схемы и визуализации
└── README.md               # Описание проекта и инструкции по запуску
```
