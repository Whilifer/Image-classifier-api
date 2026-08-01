# Image Classifier API

REST API для классификации изображений на базе модели ResNet18.

Проект создан в учебных целях как пример разработки ML-сервиса с использованием современных инструментов Python Backend и MLOps. Основная цель — показать навыки построения REST API, интеграции модели машинного обучения, контейнеризации, тестирования и организации структуры проекта.

---

# Возможности

- Классификация изображений с использованием предобученной модели ResNet18
- Возврат Top-K наиболее вероятных классов
- REST API на FastAPI
- Автоматическая загрузка модели при запуске приложения
- Dependency Injection
- Централизованная обработка исключений
- Логирование запросов и инференса
- Docker-контейнеризация
- Автоматические тесты (pytest)
- Подготовка к CI/CD через GitHub Actions

---

# Архитектура

```
                    HTTP Request
                          │
                          ▼
                   FastAPI Router
                          │
                          ▼
                 Classifier Service
                          │
                          ▼
                PyTorch ResNet18 Model
                          │
                          ▼
                  Prediction Response
```

---

# Стек технологий

| Технология | Назначение |
|------------|------------|
| Python 3.11 | Язык разработки |
| FastAPI | REST API |
| PyTorch | Инференс модели |
| Torchvision | Предобученные модели |
| Pillow | Обработка изображений |
| Pydantic Settings | Конфигурация приложения |
| Docker | Контейнеризация |
| Pytest | Тестирование |

---

# Структура проекта

```
image-classifier-api/

├── app/
│   ├── config.py
│   ├── dependencies.py
│   ├── exceptions.py
│   ├── lifespan.py
│   ├── logger.py
│   ├── main.py
│   │
│   ├── routers/
│   │   └── predict.py
│   │
│   ├── services/
│   │   ├── classifier.py
│   │   └── mlflow_service.py
│   │
│   └── utils/
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── requirements_gpu.txt
└── README.md
```

---

# Установка

```bash
git clone https://github.com/Whilifer/image-classifier-api

cd image-classifier-api

python -m venv .venv

pip install -r requirements.txt
```

---

# Запуск

```bash
uvicorn app.main:app --reload
```

После запуска документация будет доступна по адресу

```
http://127.0.0.1:8000/docs
```

---

# Запуск в Docker

Сборка образа

```bash
docker build -t image-classifier .
```

Запуск контейнера

```bash
docker run -p 8000:8000 image-classifier
```

---

# Использование API

## Проверка состояния сервиса

```
GET /predict/health
```

Ответ

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Классификация изображения

```
POST /predict/
```

Параметры

```
multipart/form-data

file=<изображение>
```

Ответ

```json
{
    "predictions": [
        {
            "class_name": "malamute",
            "confidence": 0.9113
        },
        {
            "class_name": "Eskimo dog",
            "confidence": 0.0428
        },
        {
            "class_name": "Siberian husky",
            "confidence": 0.0404
        }
    ]
}
```

---

# Тестирование

Запуск всех тестов

```bash
python -m pytest
```

Покрываются следующие сценарии:

- успешная классификация изображения;
- проверка состояния сервиса;
- запрос без файла;
- поврежденное изображение;
- загрузка файла неподдерживаемого типа.

---

# Дальнейшее развитие проекта

Планируется добавить:

- поддержку пользовательских моделей;
- ONNX Runtime;
- пакетную обработку изображений;
- MLflow Model Registry;
- мониторинг через Prometheus;
- GitHub Actions;
- авторизацию пользователей.

---

# Цель проекта

Проект разработан как демонстрация навыков создания production-подобного ML API с использованием современных инструментов Python-разработки.

Основное внимание уделено:

- организации архитектуры проекта;
- качеству кода;
- тестированию;
- контейнеризации;
- подготовке к последующему развертыванию и масштабированию.