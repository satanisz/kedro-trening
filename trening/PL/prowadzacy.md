# Scenariusz prowadzącego i rozwiązania

[Instrukcja uczestnika](warsztaty.md) i [notebook](01_cykl_zycia_modelu.ipynb) tworzą
jedną ścieżkę 4,5-godzinną. Nie wymagaj znajomości interfejsów. Przed każdym ćwiczeniem
pokaż kliknięcie na własnym ekranie, potem poproś uczestników o powtórzenie i wskazanie wyniku.

## Przygotowanie przed spotkaniem

1. Otwórz terminal w głównym katalogu repozytorium. Zainstaluj zależności:

   ```powershell
   uv sync --locked --python 3.12
   uv run python -m trening.verify_workshops
   uv run pytest -q trening/test_lifecycle.py
   ```

2. Weryfikator ma zakończyć się `PASS PL` i `PASS EN`. Kopie z wynikami znajdziesz
   w `.runtime/executed_workshop_pl.ipynb` i `.runtime/executed_workshop_en.ipynb`.
   Ich tymczasowe bazy są domyślnie usuwane. Aby zachować je do kontroli UI, dodaj
   `--keep-data`; weryfikator wypisze nadrzędny katalog.
3. Sprawdź wolne porty 8888 (JupyterLab) i 5001 (MLflow warsztatów).
   Port 5000 jest przeznaczony dla starego demonstratora, nie tego ćwiczenia.
4. Uruchom `uv run demo lab`, otwórz notebook PL i wykonaj pierwszą komórkę.
   Skopiuj wydrukowane polecenie serwera MLflow, otwórz **File → New Launcher →
   Other → Terminal**, przejdź do katalogu głównego projektu i uruchom je.
5. Nie rozsyłaj jednego `WORKSPACE` wszystkim uczestnikom. Każdy ma własną kopię
   repozytorium, własny katalog wykonania i lokalne bazy.
6. Upewnij się, że `.venv` zawiera Python 3.12, a przeglądarka pokazuje MLflow
   **Model training**. W razie zastąpienia połączeń w `conf/local` notebook zatrzyma się
   przed zapisem do obcej bazy; użyj czystej konfiguracji projektu.

## Jak prowadzić poszczególne etapy

| Etap | Pokaż | Zapytaj uczestnika | Przejdź dalej, gdy… |
| --- | --- | --- | --- |
| JupyterLab | Otwieranie pliku, Shift+Enter, `[ ]` / `[*]` / `[1]`, zapis | Co przechowuje kernel? | Widać właściwą ścieżkę Python i WORKSPACE. |
| Dane | `catalog.load`, braki, kontrakt, ID | Czy brak cechy i brak targetu są tym samym? | Wykryto duplikat oraz brak kolumny. |
| Architektura | Pliki `src`, YAML i graf `training` | Gdzie jest jedyna implementacja treningu? | Uczestnik wskazuje node i katalog. |
| Eksperymenty | Parent, child, validation, test, artefakty | Dlaczego child nie ma testu? | Widać sześć prób i konfigurację runu. |
| Wydanie | Bramkę, Model registry, alias, manifest | Czy champion automatycznie zmienia Kedro? | Wskazano konkretne wersje obu systemów. |
| Batch | Oddzielny proces i wynikową bazę | Dlaczego kolejny batch ma nowy ID? | 250 wierszy, run modelu bez zmian. |
| Monitoring | Dwa rodzaje alarmów | Co policzymy bez etykiet? | Rozróżnia drift wejść i błąd predykcji. |
| Retraining | Nowe dane, fingerprinty i nowy holdout | Dlaczego testy dwóch okresów nie wystarczą do porównania? | Oba modele oceniono na tych samych nowych rekordach. |
| Rollback | Stary manifest przy nowszym bundle | Czy techniczny rollback zawsze poprawia jakość? | Ostatni batch użył starego runu; potem przywrócono v2. |

W części architektury narysuj: **JupyterLab → KedroSession → pipeline → node →
MLflow**. Odczyt i zapis danych idą przez katalog. JupyterLab steruje eksperymentem;
logika uczenia pozostaje w `src`. Integracja z MLflow jest jawna w `training.py`;
projekt nie wymaga pluginu `kedro-mlflow`.

Kod notebooków PL i EN jest równoważny; różni się język opisu i nazwa katalogu
wykonania. Możesz prowadzić grupę mieszaną, wskazując numery sekcji, nie numery komórek.

## Oczekiwane wyniki

- Początkowe dane: 1500 treningowych, 250 scoringowych; podział 900/300/300.
- Trening: sześć child runów plus parent; `alpha` wybierane po walidacji.
- Domyślny początkowy model: `alpha=100`, test RMSE około 31 645 PLN, baseline około
  409 834 PLN. Nie oceniaj według pełnej identyczności liczb między platformami.
- W nowej bazie registry: v1 po pierwszym wydaniu, v2 po retrainingu. Model URI MLflow
  3 może mieć postać `models:/m-...`; używaj URI z raportu, nie konstruuj ścieżki na piechotę.
- Monitoring cen przesuniętych o 120 000 PLN przekracza budżet 50 000 PLN.
- Challenger na świeżym holdoucie mieści się w budżecie i poprawia RMSE o ponad 50%.
- Domyślnie 4 batche po 250 predykcji: v1 → v2 → v1 → v2. Dodatkowe uruchomienia
  zadania D zwiększają liczbę batchy i są dozwolone.
- `summary.json`: `rollback_verified=true`, `retraining_improved_rmse=true`, ostatni
  batch należy do `active_run_id`.

## Rozwiązania zadań A–F

### A. Brak kolumny

Dodaj komórkę przyciskiem **+**, wybierz typ **Code** i wykonaj:

```python
try:
    validate_frame(raw.drop(columns=["area_m2"]), schema, training=True)
except ValueError as error:
    print(error)
```

Wynik: `Missing columns: ['area_m2']`. Nie zastępuj tak uszkodzonymi danymi źródła.
Imputer obsługuje brakujące wartości w obecnej kolumnie, nie brak kolumny kontraktu.

### B. Mapa odpowiedzialności

| Szukany element | Plik |
| --- | --- |
| Walidacja i sklearn Pipeline | `src/kedro_lasso/domain.py` |
| Podział, kandydaci i logowanie MLflow | `src/kedro_lasso/training.py` |
| Składanie pipeline'ów | `src/kedro_lasso/pipeline_registry.py` |
| Nazwy wejść/wyjść node'ów | `src/kedro_lasso/pipelines/*/pipeline.py` |
| SQL i adaptery odczytu/zapisu | `conf/base/catalog.yml` |
| Cechy, siatka i podział | `conf/base/parameters.yml` |
| Transakcyjny zapis wyników | `src/kedro_lasso/datasets.py` |
| Przypięte wydanie w osobnym procesie | `trening/score_release.py` |

Scenariusz eksperymentowania bez przepisywania modelu: ustaw `runtime_params` przy
tworzeniu sesji, np. `{"tracking": {"run_name": "my-sweep"}}`. Ustawienia modelu
zmienia się w `model.alphas`, a trwałe uzgodnione zmiany trafiają do YAML. Zmianę
siatki zaplanuj przed oceną końcową, nie po obejrzeniu testu.

### C. Wybór po walidacji

```python
winner = min(initial_report["trials"], key=lambda trial: trial["rmse"])
print(winner["alpha"], initial_report["best_alpha"])
assert winner["alpha"] == initial_report["best_alpha"]
assert all("test_rmse" not in run.data.metrics for run in children)
```

Child runy służą do wyboru parametru. Parent zapisuje test po reficie train+validation.
Wielokrotne zmiany po obejrzeniu testu wymagają nowego zbioru końcowego.

### D. Scoring bez treningu

Po sekcji 5 skopiuj wypisane polecenie terminala ze ścieżką `initial.json`.
Po zakończeniu całego notebooka używaj `retrained.json`, czyli aktywnego wydania.

```python
before = len(client.search_runs([experiment_id]))
run_release(retrained_release)  # Use initial_release if section 7 has not run yet.
assert len(client.search_runs([experiment_id])) == before
display(warehouse_query(history_sql))
```

W SQL znajdziesz nowy `batch_id`, ten sam `run_id` i 250 wierszy. Samo ponowne
wykonanie nie aktualizuje poprzedniego batcha.

### E. Drift i jakość

Przesunięcie `area_m2` wywołuje alarm rozkładu. Do zmierzenia RMSE potrzebujesz
rzeczywistych cen. Zmiana relacji ceny do cech może pogorszyć RMSE bez dużej zmiany
rozkładów wejść. Heurystyka średnich i braków nie wykrywa wszystkich rodzajów driftu.
Nowe etykiety trzeba łączyć z zachowanymi predykcjami/ID i czasem ich wykonania.
W tym ćwiczeniu dane monitoringowe są generowane lokalnie; nie ma integracji z systemem etykiet.

### F. Przypięty rollback

Alias jest zmienny. Manifest przypina `registry_version` i `kedro_bundle_version`.
`score_release.py` sprawdza powiązanie i sumę kontrolną, następnie przekazuje
`load_versions` do Kedro. Dlatego stary manifest używa v1 również po wytrenowaniu v2.
Wynik biznesowy może się pogorszyć, jeśli poprzedni model nie pasuje do nowego reżimu.
Rollback i promocja wymagają decyzji operacyjnej, a nie samego wyboru najmniejszego numeru wersji.

## Rozwiązanie zadania końcowego i kryteria zaliczenia

Po wykonaniu całego notebooka:

```python
latest = warehouse_query(history_sql).iloc[-1]
manifest = json.loads(retrained_release.read_text(encoding="utf-8"))
assert latest["run_id"] == manifest["run_id"]
display(
    pd.Series(
        {
            "batch_id": latest["batch_id"],
            "run_id": manifest["run_id"],
            "registry_version": manifest["registry_version"],
            "kedro_bundle_version": manifest["kedro_bundle_version"],
        }
    )
)
print(f'uv run python -m trening.score_release --release "{retrained_release}"')
```

Artefakty konfiguracji i ID podziałów są w parent runie w MLflow. Gdy challenger
nie przejdzie bramki, pozostaw bieżące wydanie, zbadaj przyczynę oraz uzgodnij reakcję
biznesową; nie promuj go automatycznie tylko dlatego, że stary model ma alarm.

Przyznaj po 1 punkcie za: kontrakt i rozłączne dane; wybór po walidacji; wskazanie
runu, artefaktu i wersji; batch bez treningu; poprawne rozróżnienie alarmów; retraining
na oddzielnych danych i holdoucie; dowód rollbacku; świadome ograniczenia produkcyjne.
Zaliczenie: **6/8** oraz obowiązkowo batch bez treningu i identyfikowalny rollback.

## Omówienie produkcji i wycofania modelu

Przed realnym wdrożeniem omów: walidację czasową/grupową; monitoring etykiet i jakości
w segmentach; niezmienny pakiet kodu i zależności; prawa do danych; rejestr decyzji;
uwierzytelnianie; PostgreSQL i magazyn obiektowy; migracje, orkiestrację, backupy oraz retencję.
Repozytorium nie implementuje tych funkcji produkcyjnych w ramach tego ćwiczenia.

Wycofanie modelu: zatrzymaj przyszłe batch joby lub przepnij konsumentów, zapisz
powód i właściciela decyzji, zachowaj wymagane dowody, a usuwanie artefaktów uzależnij
od uzgodnionej retencji. W warsztacie v1 pozostaje `rollback_available`, v2 jest `active`.
Nie kasuj v1, ponieważ ćwiczenie wymaga odtworzenia poprzedniego wydania.

## Zakończenie

Poproś o zapis notebooków i zachowanie manifestów. Zatrzymaj własny MLflow przez
Ctrl+C w jego terminalu, a następnie JupyterLab przez Ctrl+C w terminalu launchera.
Nie kończ procesów innych osób. Przy ponownym starcie notebooka uruchom MLflow na
nowej wypisanej ścieżce. Szczegółowe problemy i odzyskiwanie są w instrukcji uczestnika.
