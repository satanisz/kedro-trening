# Model lifecycle workshops / Warsztaty cyklu życia modelu

Two equivalent, guided workshops for beginners using this repository's Lasso model,
Kedro pipelines, JupyterLab, MLflow Tracking and Model Registry, and separate SQLite
source and output databases. Here, “flowML” means **MLflow**.

| Language / Język | Start here / Zacznij tutaj | Executable notebook / Notebook | Instructor / Prowadzący |
| --- | --- | --- | --- |
| Polski | [Instrukcja krok po kroku](PL/warsztaty.md) | [Cykl życia modelu](PL/01_cykl_zycia_modelu.ipynb) | [Scenariusz i rozwiązania](PL/prowadzacy.md) |
| English | [Click-by-click guide](EN/workshop.md) | [Model lifecycle](EN/01_model_lifecycle.ipynb) | [Facilitation and solutions](EN/instructor.md) |

## Requirements / Wymagania

Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and a
browser. Participants need basic Python and DataFrame knowledge; previous Kedro,
MLflow, or JupyterLab experience is not required. Docker is optional and is not used
in the workshop's main path. Allow **4 hours 30 minutes**, including two breaks.

## Installation / Instalacja

From the repository root / Z głównego katalogu repozytorium:

```powershell
uv sync --locked --python 3.12
```

## Configuration / Konfiguracja

Every execution from the beginning creates its own timestamped directory under
`.runtime/trening/pl/` or `.runtime/trening/en/`. The notebook selects this directory
before reloading Kedro. It uses local SQLite for MLflow tracking and registry metadata,
local files for model artifacts, and another SQLite file for predictions.

Każde wykonanie od początku tworzy nowy katalog pod `.runtime/trening/pl/` lub
`.runtime/trening/en/`. Notebook ustawia go przed odświeżeniem Kedro. Metadane MLflow
i rejestru są w lokalnym SQLite, artefakty w plikach, a predykcje w oddzielnym SQLite.

`KEDRO_WORKSHOP_ROOT` optionally changes the parent of these execution directories.
Do not run writing pipelines concurrently in the same execution directory.

## Quick start / Start

```powershell
uv run demo lab
```

Open [JupyterLab](http://127.0.0.1:8888/lab?token=kedro-demo). In the left file browser,
double-click `trening`, then `PL` or `EN`, and open the notebook. Follow the guide in
that directory. Run one code cell at a time with **Shift+Enter**. Section 0 prints the
exact MLflow server command for this execution; paste it into a second terminal and
open [the workshop MLflow UI](http://127.0.0.1:5001). Port 5000 belongs to the original demo.

Otwórz [JupyterLab](http://127.0.0.1:8888/lab?token=kedro-demo). Po lewej kliknij dwukrotnie
`trening`, następnie `PL`, a potem notebook. Czytaj `warsztaty.md` i wykonuj komórki
przez **Shift+Enter**. Sekcja 0 wypisze polecenie serwera MLflow: wklej je do drugiego
terminala i otwórz [MLflow warsztatów](http://127.0.0.1:5001). Port 5000 należy do starego demo.

## What you will run / Przebieg

```mermaid
flowchart LR
    goal[Goal and release criteria] --> data[(Source SQLite)]
    data --> explore[JupyterLab exploration]
    explore --> kedro[Kedro validation and training]
    kedro --> tracking[MLflow experiments]
    tracking --> gate[Quality gate]
    gate --> registry[Registry and pinned release]
    registry --> batch[Kedro batch in a separate process]
    batch --> output[(Output SQLite)]
    output --> monitor[Inputs and delayed-label quality]
    monitor --> retrain[Kedro retraining]
    retrain --> gate
    registry --> rollback[Rollback with previous manifest]
    rollback --> batch
```

The executable path includes model registration, an alias, a release manifest, scoring
with a pinned Kedro bundle, monitoring, independent evaluation of a retrained model,
and a verified rollback. It reuses training and scoring from `src/kedro_lasso/`.
The guides include exercises, expected results, UI actions, a glossary, and recovery steps.

Ścieżka wykonywalna obejmuje rejestrację, alias, manifest wydania, scoring przypiętą
wersją Kedro, monitoring, niezależną ocenę ponownie wytrenowanego modelu i sprawdzony
rollback. Trening i scoring pochodzą z `src/kedro_lasso/`. Instrukcje zawierają zadania,
oczekiwane wyniki, kliknięcia, słownik i pomoc przy błędach.

## Verification / Weryfikacja

```powershell
uv run python -m trening.verify_workshops
uv run pytest -q trening/test_lifecycle.py
uv run ruff check trening
uv run ruff format --check trening
```

The verifier runs both notebooks in disposable workspaces and saves executed copies as
`.runtime/executed_workshop_pl.ipynb` and `.runtime/executed_workshop_en.ipynb`.
Add `--keep-data` to retain verification databases for UI inspection. No web service
is started by the verifier. Source notebooks are committed without outputs.

Weryfikator wykonuje oba notebooki w tymczasowych katalogach i zapisuje kopie z wynikami
jako `.runtime/executed_workshop_pl.ipynb` i `.runtime/executed_workshop_en.ipynb`.
Opcja `--keep-data` zachowuje bazy do obejrzenia w UI. Weryfikator nie uruchamia serwerów.
Notebooki źródłowe pozostają bez wyników.

## Boundaries / Granice

Deployment means a **local batch**, not a hosted prediction API. Promotion and monitoring
are manual exercises. The input monitor is a teaching heuristic. The material explains
team deployment, time-aware evaluation, security, orchestration, retention, and retirement
as further work, rather than claiming that this local stack implements them.

Wdrożenie oznacza **lokalny batch**. Promocja i monitoring są uruchamiane ręcznie,
a monitor wejść to heurystyka dydaktyczna. Materiały omawiają wdrożenie zespołowe,
walidację czasową, bezpieczeństwo, orkiestrację, retencję i wycofywanie modeli jako
dalszy rozwój tego przykładu.
