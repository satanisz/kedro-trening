# Scenariusz demonstracji

## 1. Pierwszy przebieg

```powershell
uv sync --locked --python 3.12
uv run demo stack --with-viz
```

Launcher wykonuje pipeline, a następnie uruchamia narzędzia. Po starcie zobaczysz URL-e usług i lokalizację logów. Daj usługom chwilę na inicjalizację; w razie błędu sprawdź `.runtime/logs/`.

## 2. JupyterLab: klient projektu

Otwórz [JupyterLab](http://127.0.0.1:8888/lab?token=kedro-demo), potem `notebooks/01_kedro_mlflow_lab.ipynb`. Użyj kernela środowiska projektu. `demo lab` ustawia go domyślnie; notebook ma też jawne wczytanie rozszerzenia Kedro.

Wykonuj komórki kolejno albo wybierz **Run → Run All Cells**. Obserwuj:

1. Odczyt SQLite przez `catalog.load('raw_training')`.
2. Ponowne użycie `validate_frame()` z `src/` podczas eksploracji.
3. Wykonanie pełnego pipeline’u w nowej sesji z inną siatką `alpha`.
4. Odczyt raportu, porównanie regularyzacji i współczynników modelu.
5. Odczyt drugiej bazy przez katalog.
6. Scoring bez treningu i odtworzenie modelu bez Kedro przez MLflow.

Po uruchomieniu sekcji treningowej przejdź do MLflow, aby obejrzeć nowe runy. Po zmianie kodu lub YAML użyj `%reload_kedro` i nowej sesji do wykonania.

## 3. MLflow: czego szukać

Otwórz [MLflow](http://127.0.0.1:5000), przełącz **Model training**, wybierz eksperyment `housing-lasso` i zakładkę **Runs**. Domyślny widok GenAI pokazuje inne narzędzia niż potrzebne do regresji.

- `lasso-sweep` to przebieg z konfiguracji YAML; `notebook-sweep` pochodzi z notebooka.
- Każdy ma podrzędne runy `alpha=...`. Porównaj `validation_rmse` i liczbę niezerowych współczynników.
- Nadrzędny run ma `best_alpha`, `test_rmse`, `test_mae`, `test_r2` i metryki baseline.
- W artefaktach znajdziesz konfigurację, fingerprinty, ID podziałów, współczynniki i raport. Model jest zalogowany jako `lasso`.

Domyślny wynik dla danych `seed=42` to `alpha=100`, testowy RMSE około 31,6 tys. PLN i R² około 0,994. Porównuj go z baseline i pamiętaj, że dane są syntetyczne. Inna siatka lub podział może zmienić wybór i metryki.

## 4. Baza wynikowa i niezależny scoring

W drugim terminalu:

```powershell
uv run demo inspect
uv run kedro run --pipeline scoring
uv run demo inspect
```

Powstaje kolejna partia 250 predykcji dla tego samego modelu; nie powstają nowe runy treningowe. W SQLite możesz sprawdzić:

```sql
SELECT b.batch_id, b.run_id, b.n_rows, m.best_alpha, m.test_rmse
FROM prediction_batches b
JOIN model_runs m ON m.run_id = b.run_id
ORDER BY b.created_at DESC;

SELECT p.property_id, p.prediction, b.run_id
FROM predictions p
JOIN prediction_batches b ON b.batch_id = p.batch_id
ORDER BY b.created_at DESC, p.property_id
LIMIT 10;
```

Lokalna baza to `data/warehouse/warehouse.db`. Przy Compose jest we współdzielonym wolumenie pod `/app/data/warehouse/warehouse.db`.

## 5. Opcjonalny graf

W [Kedro-Viz](http://127.0.0.1:4141) wybierz pipeline domyślny, `training` albo `scoring`. Zwróć uwagę, że scoring zależy od zapisanego modelu i danych wejściowych, a nie od notebooka. Węzły mają tagi; katalog definiuje warstwy raw/intermediate/models/scoring/reporting.

## 6. Drobna zmiana architektury przez konfigurację

```powershell
uv run kedro run --env quick --pipeline training
uv run kedro run --pipeline scoring
```

Środowisko `quick` zmniejsza siatkę do dwóch wartości `alpha`. Model jest zapisany jako nowa wersja, a scoring ładuje najnowszą. Potem przywróć pełny trening przez zwykłe `uv run kedro run`.

Zakończenie lokalnego stacka: `Ctrl+C` w jego terminalu. Dla Compose użyj `docker compose down`; dane pozostają w wolumenie.
