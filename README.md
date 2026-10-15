# Kedro + JupyterLab + MLflow + SQLite

Działający przykład przepływu ML: źródłowa baza SQLite → pipeline Kedro → eksperymenty Lasso w MLflow → predykcje w osobnej bazie SQLite. Notebook korzysta z tego samego katalogu, konfiguracji i kodu co uruchomienie wsadowe. Kedro-Viz jest opcjonalny.

## Wymagania

Python 3.12 i [uv](https://docs.astral.sh/uv/getting-started/installation/). Wariant kontenerowy wymaga działającego Docker Engine oraz Docker Compose.

## Instalacja

Uruchom w katalogu projektu:

```powershell
uv sync --locked --python 3.12
```

Dokładne wersje zależności zapisuje `uv.lock`. Stack zawiera Kedro 1.7.0, Kedro-Datasets 9.6.0, Kedro-Viz 12.5.0, JupyterLab 4.6.4 i MLflow 3.17.1.

## Konfiguracja

Domyślna konfiguracja działa bez zmian i bez pobierania danych. `conf/base/parameters.yml` definiuje cechy, podział danych i siatkę `alpha`. `conf/base/catalog.yml` definiuje wejścia SQL i wyjścia; `conf/base/globals.yml` — ścieżki i URI.

Opcjonalne zmienne środowiskowe: `KEDRO_DEMO_DATA_DIR` (bezwzględny katalog danych), `MLFLOW_TRACKING_URI` (zewnętrzny serwer śledzenia) i `KEDRO_DEMO_JUPYTER_TOKEN` (token lokalnego JupyterLab). Nadpisania projektu umieszczaj w `conf/local/`.

## Pierwsze uruchomienie

```powershell
uv run demo stack
```

Komenda tworzy źródło, jeśli go brakuje, wykonuje pipeline i uruchamia MLflow oraz JupyterLab. Działa w bieżącym terminalu; `Ctrl+C` zatrzymuje usługi. Logi usług są w `.runtime/logs/`.

| Narzędzie | Adres |
| --- | --- |
| JupyterLab | [Otwórz laboratorium](http://127.0.0.1:8888/lab?token=kedro-demo) |
| MLflow | [Otwórz eksperymenty](http://127.0.0.1:5000) |
| Kedro-Viz, po włączeniu | [Otwórz graf](http://127.0.0.1:4141) |

W JupyterLab otwórz [polski notebook](notebooks/01_kedro_mlflow_lab.ipynb) albo [English notebook](notebooks/02_kedro_mlflow_lab_en.ipynb) i wybierz **Run → Run All Cells**. W MLflow przełącz **Model training**, wybierz eksperyment **housing-lasso**, zakładkę **Runs** i rozwiń uruchomienie nadrzędne, żeby porównać wartości `alpha`. Link do JupyterLab używa domyślnego tokenu demonstracyjnego; po jego zmianie użyj adresu wypisanego przez launcher.

Z opcjonalnym grafem:

```powershell
uv run demo stack --with-viz
```

Jeżeli model już istnieje i chcesz jedynie otworzyć narzędzia:

```powershell
uv run demo stack --skip-run --with-viz
```

## Uruchomienia wsadowe

```powershell
uv run demo seed
uv run kedro run
uv run demo inspect
```

Źródło zawiera 1500 syntetycznych nieruchomości z ceną i 250 bez ceny. Pipeline porównuje sześć wartości `alpha`, wybiera model na walidacji i ocenia go na oddzielnym teście. MLflow przechowuje parametry, metryki, konfigurację, fingerprinty danych oraz model z imputerem i skalowaniem. Wynikowy SQLite przechowuje historię modeli, prób i partii predykcji.

| Zadanie | Polecenie |
| --- | --- |
| Tylko przygotowanie danych treningowych | `uv run kedro run --pipeline prepare` |
| Tylko trening i zapis modelu | `uv run kedro run --pipeline training` |
| Predykcja istniejącym modelem | `uv run kedro run --pipeline scoring` |
| Krótka siatka eksperymentów | `uv run kedro run --env quick --pipeline training` |
| Zmiana seeda podziału | `uv run kedro run --params split.random_state=123` |

W osobnych terminalach można uruchomić `uv run demo mlflow`, `uv run demo lab` i `uv run demo viz`.

## Docker Compose

```powershell
docker compose up --build
```

Compose uruchamia seed, serwer MLflow, pipeline i JupyterLab w tej kolejności. Profil `viz` dodaje wizualizację:

```powershell
docker compose --profile viz up --build
```

Kolejne wykonanie wsadowe i zatrzymanie usług:

```powershell
docker compose run --rm pipeline kedro run --env quick
docker compose down
```

Dane kontenerów są w wolumenie `kedro-lasso_demo-data`; lokalne uruchomienie używa katalogu `data/`. To osobne środowiska. `down` zachowuje wolumen. Kod, konfiguracja i notebooki są montowane z projektu, więc zmiany są widoczne bez przebudowy obrazu. Zmiana zależności wymaga ponownego builda.

## Dokumentacja i testy

[Architektura i schemat baz](docs/architecture.md) opisują granice komponentów, transakcje, zmiany konfiguracji i ograniczenia demonstratora. [Scenariusz demonstracji](docs/walkthrough.md) prowadzi przez notebook i MLflow.

```powershell
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/verify_notebook.py
```

Testy integracyjne i weryfikator notebooka używają oddzielnych baz tymczasowych. Notebook po weryfikacji zapisuje się jako `.runtime/executed_notebook.ipynb`. Workflow CI uruchamia te same kontrole na Windows i Linux.

To demonstrator architektury do pracy lokalnej: SQLite zakłada małą liczbę zapisujących procesów, a usługi są udostępniane na interfejsie loopback. Szczegóły przejścia do środowiska współdzielonego opisuje dokumentacja architektury.
