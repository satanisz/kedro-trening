# Warsztaty: pełny cykl życia modelu, krok po kroku

**Czas:** 4 godziny 30 minut z przerwami. **Odbiorcy:** osoby znające podstawy Pythona,
które zaczynają pracę z Kedro, MLflow i JupyterLab. Wszystkie dane są syntetyczne.
Notebook to kompletny przykład; zadania A–F wykonuj w dodatkowych komórkach.
[Materiały prowadzącego z rozwiązaniami](prowadzacy.md) otwórz po próbie samodzielnej pracy.

Po warsztatach umiesz pobrać dane z katalogu Kedro, uruchomić trening z notebooka,
porównać eksperymenty, wydać konkretną wersję, wykonać batch poza notebookiem oraz
zareagować na zmianę danych, pogorszenie jakości i awarię wdrożenia.

## 1. Co robią poszczególne narzędzia

| Pojęcie | Znaczenie w ćwiczeniu |
| --- | --- |
| JupyterLab | Strona w przeglądarce do otwierania plików, wykonywania komórek i oglądania wyników. |
| Notebook | Plik `.ipynb` z opisami i komórkami kodu wykonywanymi przez kernel. |
| Kernel | Proces Pythona przechowujący zmienne notebooka. Restart usuwa zmienne z pamięci. |
| Kedro node | Funkcja Pythona z nazwanymi wejściami i wyjściami. |
| Pipeline | Graf zależności między node'ami, uruchamiany przez Kedro. |
| Data Catalog | Mapa nazwa danych → sposób odczytu/zapisu. Tu m.in. SQLite, Parquet i pickle. |
| Parametry | Ustawienia YAML, np. lista cech, podział danych i wartości `alpha`. |
| MLflow experiment | Grupa powiązanych uruchomień. Tu `workshop-lifecycle`. |
| MLflow run | Jedno zapisane wykonanie z parametrami, metrykami i artefaktami. |
| Artefakt | Plik związany z runem, np. model, konfiguracja lub raport. |
| Model Registry | Rejestr wersji zalogowanych modeli, oddzielny od listy runów. |
| Alias | Zmienna nazwa wskazująca wersję, np. `champion`. |
| Manifest wydania | JSON przypinający wersje, powiązania i wynik bramki jakości. |
| Batch / scoring | Jednorazowe przewidywanie cen dla całej tabeli bez treningu. |
| Baseline | Prosty punkt odniesienia; tutaj model przewidujący średnią cenę. |
| RMSE / MAE / R² | Pierwiastek średniego błędu kwadratowego, średni błąd bezwzględny i miara dopasowania; mniejsze RMSE/MAE jest lepsze. |
| Drift / retraining / rollback | Zmiana danych lub relacji / ponowny trening / powrót do poprzedniego wydania. |

Model Lasso to regresja liniowa z karą za duże współczynniki. Większe `alpha` oznacza
silniejszą regularyzację; część współczynników może spaść do zera. Nie zakładamy, że
największe lub najmniejsze `alpha` daje najlepszy model.

## 2. Plan spotkania

| Czas od startu | Działanie | Dowód ukończenia |
| --- | --- | --- |
| 00:00–00:20 | Instalacja i JupyterLab | Notebook otwarty, właściwy Python. |
| 00:20–00:50 | Cel, dane, kontrakt | 1500/250 rekordów; kontrolowany błąd. |
| 00:50–01:25 | Architektura i Kedro w notebooku | Rozłączne zbiory 900/300/300; graf. |
| 01:25–01:35 | Przerwa | |
| 01:35–02:10 | Eksperymenty i MLflow | Parent + sześć child runów. |
| 02:10–02:40 | Bramka i rejestr | Wersja modelu, `champion`, manifest. |
| 02:40–03:05 | Wdrożenie batch | 250 predykcji, bez nowego treningu. |
| 03:05–03:15 | Przerwa | |
| 03:15–03:45 | Monitoring | Osobno alarm wejść i alarm RMSE. |
| 03:45–04:15 | Retraining i rollback | Nowy model oraz batch ze starej wersji. |
| 04:15–04:30 | Zadanie końcowe i omówienie | Powiązanie ostatniego batcha z runem. |

## 3. Przygotuj komputer i uruchom JupyterLab

1. W Windows otwórz **Eksplorator plików** (`Win+E`) i przejdź do katalogu projektu.
   Na tym komputerze jest to `C:\Users\stani\Projects\kedro_trening`.
2. Kliknij pasek adresu, wpisz `powershell` i naciśnij **Enter**. Terminal powinien
   pokazywać katalog z plikiem `pyproject.toml`.
3. Wklej polecenie poniżej i naciśnij **Enter**. Instalacja wymaga sieci przy pierwszym użyciu.
   Jeśli polecenie `uv` nie istnieje, zainstaluj je według
   [instrukcji uv](https://docs.astral.sh/uv/getting-started/installation/), otwórz nowy terminal i ponów.

   ```powershell
   uv sync --locked --python 3.12
   ```

4. Po zakończeniu, gdy wróci znak zachęty terminala, uruchom:

   ```powershell
   uv run demo lab
   ```

5. Pozostaw ten terminal otwarty. W przeglądarce otwórz
   [JupyterLab](http://127.0.0.1:8888/lab?token=kedro-demo).
   Przy własnym tokenie użyj adresu wypisanego przez launcher.
6. Po lewej stronie JupyterLab znajdziesz listę plików. Kliknij dwukrotnie **trening**,
   następnie **PL**, następnie **01_cykl_zycia_modelu.ipynb**.
7. W tym samym folderze możesz kliknąć prawym przyciskiem `warsztaty.md` i wybrać
   **Open With → Markdown Preview**, aby czytać instrukcję. Wróć do karty notebooka.
8. Jeżeli pojawi się okno wyboru kernela, wybierz Python ze środowiska projektu.
   Gdy nie wiesz który: wybierz dostępny Python i sprawdź `Python:` po pierwszej komórce.
   Ścieżka powinna wskazywać `.venv` tego projektu. Jeśli nie wskazuje, użyj
   **Kernel → Change Kernel…** i wybierz kernel projektu zainstalowany przez `demo lab`.

Na Linux/macOS otwórz terminal w katalogu repozytorium i wykonaj te same polecenia `uv`.

## 4. Naucz się wykonywać komórki

1. Komórka z opisem nie wykonuje Pythona. Komórka z kodem ma pole `[ ]` po lewej.
2. Kliknij wewnątrz pierwszej komórki kodu w sekcji **0** i naciśnij **Shift+Enter**.
   Alternatywnie użyj **Run → Run Selected Cell**.
3. Podczas pracy zobaczysz `[*]`. Poczekaj, aż pojawi się numer, np. `[1]`.
4. Wynik pojawi się pod kodem. Nie wpisuj wypisanych wyników z powrotem do komórki.
5. Skopiuj ścieżkę `WORKSPACE` do swoich notatek. Każdy świeży start ma inną ścieżkę.
6. Wykonaj komórkę sekcji **1**. Oczekuj listy pipeline'ów `prepare`, `training`,
   `scoring`, `__default__` oraz nazw datasetów, w tym `raw_training`.
7. Kolejne komórki wykonuj po kolei, od góry. Nie naciskaj wielokrotnie **Shift+Enter**,
   gdy kod jeszcze działa. Pełny przykład można wykonać przez **Run → Run All Cells**,
   ale na zajęciach zatrzymuj się przy opisach i zadaniach.
8. Aby dodać komórkę zadania: kliknij istniejącą, kliknij **+** nad notebookiem
   (podpowiedź **Insert a cell below**), ustaw typ **Code** w rozwijanej liście,
   wpisz kod i uruchom **Shift+Enter**. Nie nadpisuj komórek bazowej ścieżki.
9. Zapisuj pracę przez **File → Save Notebook** lub **Ctrl+S**.

## 5. Dane, kontrakt i architektura — sekcja 2 notebooka

1. Wykonaj komórkę z `catalog.load`. Pod nią zobaczysz tabelę, statystyki, odsetek braków
   i wykres ceny względem powierzchni. Oczekuj 1500 rekordów treningowych i 250 scoringowych.
2. W tabeli znajdź `property_id`, `price_pln` i `area_m2`. Ceny scoringu nie są dostępne.
3. Wykonaj komórkę **Kontrolowany błąd**. Komunikat `EXPECTED VALIDATION ERROR` jest
   wynikiem pozytywnym: kontrakt wykrył powtórzony identyfikator.
4. Wykonaj **zadanie A**: w dodatkowej komórce usuń `area_m2` i wywołaj `validate_frame`.
   Złap `ValueError`, tak jak w przykładzie. Oczekuj wskazania brakującej kolumny.
5. Wykonaj komórkę podziału. Oczekuj 900 rekordów train, 300 validation, 300 test.
   Asercje sprawdzają, że żadne ID nie należy do dwóch zbiorów.
6. Aby obejrzeć kod (**zadanie B**), kliknij ikonę folderu po lewej, następnie symbol
   katalogu głównego w pasku ścieżki. Otwórz **src → kedro_lasso**, a potem dwukrotnie:
   `domain.py`, `training.py` i `pipeline_registry.py`. Otworzą się karty edytora.
7. W `domain.py` znajdź `SimpleImputer`, `StandardScaler`, `Lasso`. W `training.py`
   znajdź `mlflow.start_run`. W `pipeline_registry.py` zobacz składanie pipeline'ów.
   Wróć do karty notebooka. Niczego nie musisz zmieniać w tych plikach.

**Zatrzymaj się i odpowiedz:** dlaczego uzupełnienie braków na całym zbiorze przed
podziałem mogłoby ujawnić informacje ze zbioru testowego?

## 6. Trening i MLflow — sekcja 3

1. Uruchom komórkę z `training_session.run(pipeline_name="training")` i poczekaj na koniec.
   Logi Kedro oraz ostrzeżenia o serializacji własnego modelu nie oznaczają niepowodzenia.
2. Pod komórką zobaczysz sześć prób `alpha`, metryki Lasso i baseline, wersję Kedro,
   `MLflow run` i `Model URI`. Zachowaj `run_id` jako identyfikator tego modelu.
3. Wróć do wyniku sekcji **0**. Zaznacz i skopiuj **całe** polecenie zaczynające się od
   `uv run python -m mlflow server`. Zawiera ścieżkę Twojego wykonania, więc nie używaj
   polecenia sąsiada ani komendy ze starego uruchomienia notebooka.
4. W JupyterLab wybierz **File → New Launcher**, potem kafelek **Terminal** w sekcji
   **Other**. W nowej karcie terminala wpisz
   `pwd` i naciśnij Enter. Jeśli nie jesteś w katalogu projektu, wpisz
   `cd "C:\Users\stani\Projects\kedro_trening"` (lub własną ścieżkę).
5. Wklej skopiowane polecenie i naciśnij **Enter**. Pozostaw terminal otwarty.
   Gdy proces zgłosi gotowość, otwórz [MLflow warsztatów](http://127.0.0.1:5001).
6. W lewym górnym wyborze obszaru kliknij **Model training**, jeśli widzisz widok GenAI.
   W lewym menu wybierz **Experiments** i na liście kliknij **workshop-lifecycle**.
7. W zakładce **Runs** znajdź **initial-sweep**. Kliknij strzałkę po lewej, aby rozwinąć
   sześć podrzędnych runów `alpha=...`. Jeżeli po chwili się nie pojawiły, odśwież stronę.
8. Kliknij nazwę jednego `alpha=...`. W szczegółach obejrzyj `alpha` oraz `validation_rmse`.
   Wróć przyciskiem przeglądarki do listy. Otwórz **initial-sweep** i znajdź `best_alpha`,
   `test_rmse`, `baseline_test_rmse` i zapisany model `lasso`.
9. W szczegółach parent runu kliknij **Artifacts**. Klikaj pliki `configuration.json`,
   `split_ids.json`, `data_fingerprints.json` i `training_report.json`, aby zobaczyć podgląd.
10. **Zadanie C:** zapisz zwycięskie `alpha` i wskaż, w których runach są wyniki testu.
    Wróć do JupyterLab, uruchom komórkę z `client.search_runs` i porównaj tabelę z UI.

**Oczekiwany sens wyniku:** sześć child runów ma walidację; parent przechowuje ocenę
końcową. Przykład domyślny wybiera `alpha=100` i ma RMSE około 31,6 tys. PLN, ale
kryterium poprawności to działające kontrole i wcześniej ustalone progi, nie identyczny tekst logu.

## 7. Bramka, rejestr i promocja — sekcja 4

1. Uruchom komórkę `quality_gate`. Oczekuj `passed: True`. Kopia raportu ze sztucznym
   RMSE 200 000 jest odrzucana. Nie obchodź nieudanej bramki zmianą flagi w JSON.
2. Ta sama komórka rejestruje zaakceptowany model i ustawia alias `champion`.
   W nowej bazie jest to wersja rejestru **1**. Wersja Kedro ma postać znacznika czasu;
   to inny system wersjonowania.
3. W MLflow wybierz po lewej **Model registry** (nie zakładkę **Models** eksperymentu).
   Przy pierwszej wizycie zamknij okno wprowadzające przyciskiem **Try it now**.
   Kliknij **workshop-housing-lasso**, potem wiersz
   wersji **1**. Zobacz źródłowy run, alias `champion` i tag `kedro_bundle_version`.
   W tych warsztatach aliasy zmienia notebook, więc nie zmieniaj ich ręcznie w UI.
4. Wróć do notebooka i uruchom komórkę **Zapisz decyzję o wydaniu**.
   Oczekuj ścieżki do `releases/initial.json` i jego zawartości.
5. W parent runie MLflow otwórz **Artifacts → releases → initial.json**.
   Porównaj `run_id`, `registry_version`, `kedro_bundle_version` i wynik bramki.

**Sprawdzenie:** rejestracja tworzy wersję; promocja wskazuje wersję aliasem; wdrożenie
uruchamia pracę z konkretnym artefaktem. Są to trzy różne działania.

## 8. Wdrożenie wsadowe — sekcja 5

1. Wykonaj komórkę `run_release(initial_release)`. Wywoła osobny proces, który użyje
   istniejącego pipeline'u `scoring` i konkretnej wersji bundle z manifestu.
2. Wykonaj komórkę zapytań do warehouse. Zobaczysz jeden batch z `n_rows=250`, jego
   `run_id` i przykładowe predykcje. Asercja porówna model z rejestru z modelem Kedro.
3. Sprawdź listę **Runs** w MLflow. Nowego `initial-sweep` nie powinno być.
4. **Zadanie D:** skopiuj wypisane `Terminal command: uv run python -m trening.score_release…`.
   Otwórz **File → New Launcher → Other → Terminal**, sprawdź katalog `pwd`,
   wklej polecenie i Enter.
   Poczekaj na powrót znaku zachęty.
5. Wróć do notebooka, dodaj nową komórkę `display(warehouse_query(history_sql))` i ją uruchom.
   Jest nowy batch, ale ten sam `run_id`. Każdy batch zawiera 250 predykcji.

Zapis tej samej przygotowanej partii jest idempotentny w adapterze SQL, ale ponowne
uruchomienie scoringu tworzy **nowy** batch ID. Nie myl tych dwóch zachowań.

## 9. Monitoring — sekcja 6

1. Uruchom `shifted_inputs`. W tabeli znajdź `area_m2`: `alert` ma być `True`.
   Zmiana jest tylko w pamięci; nie uszkadza bazy.
2. Uruchom komórkę **Nadeszły ceny**. Wypisane RMSE starego modelu przekroczy 50 000 PLN.
   Generator zasymulował ceny wyższe o 120 000 PLN przy podobnych cechach.
3. W MLflow kliknij **Experiments → workshop-monitoring → labelled-future-period**.
   Obejrzyj `observed_rmse`, tag `model_run_id` i artefakt `input_monitor.json`.
4. **Zadanie E:** wyjaśnij dwie sytuacje: zmieniły się cechy, ale cen jeszcze nie ma;
   cechy wyglądają podobnie, ale po otrzymaniu cen widać pogorszenie jakości.

Alarm wejść nie jest dowodem spadku jakości. Nie uruchamiaj automatycznej promocji
wyłącznie na podstawie takiego alarmu. Monitoring bez etykiet i z etykietami to inne kontrole.

## 10. Retraining, nowy holdout i rollback — sekcje 7–9

1. Wykonaj komórkę **7**. Zachowa stare dane w Parquet i zastąpi tabele tylko w Twojej
   bazie warsztatowej. Nowy trening ma nazwę `retrained-sweep` i inny fingerprint.
2. Wykonaj **Challenger kontra champion**. Porównanie odbywa się na nowym holdoucie:
   inne ID i seed niż trening i monitoring. Nie porównuj jakości modeli jedynie po
   metrykach ich różnych testów wewnętrznych.
3. Oczekuj, że nowy model zmieści się w 50 000 PLN i obniży RMSE o co najmniej 50%
   względem starego na tym samym holdoucie. Wtedy powstanie wersja rejestru **2**,
   `releases/retrained.json`, nowy `champion` i batch z runem v2.
4. W MLflow otwórz **Model registry → workshop-housing-lasso** i odśwież stronę. Zobacz wersje
   1 i 2 oraz aliasy. `previous` wskazuje v1, a `champion` i `candidate` wskazują v2.
5. Uruchom sekcję **8**. To kontrolowany rollback: stary manifest powoduje scoring
   modelem v1, chociaż najnowszy bundle w katalogu jest v2.
6. W tabeli znajdź ostatni `run_id` i porównaj z `initial_report['run_id']`.
   Komunikat `Rollback verified` potwierdza również zgodność samych predykcji.
7. **Zadanie F:** wyjaśnij, dlaczego alias i przypięty manifest trzeba traktować osobno.
   W nowym reżimie cen rollback do v1 pogarsza jakość; to demonstracja operacyjna.
8. Uruchom sekcję **9**. Przywróci zaakceptowaną v2 i zapisze `summary.json`.
   Bez dodatkowych uruchomień zadań są **4 batche po 250 predykcji**. Po zadaniu D
   batchy jest więcej — to poprawne. Ostatni ma wskazywać `active_run_id` z podsumowania.

## 11. Zadanie końcowe — 15 minut

W nowych komórkach przygotuj krótką notatkę oraz zapytanie/tabelę z odpowiedziami:

1. Jaki `run_id`, wersja rejestru i wersja Kedro odpowiadają ostatniemu batchowi?
2. Gdzie znajduje się konfiguracja eksperymentu oraz lista ID podziałów?
3. Jak wykonasz kolejny batch bez treningu? Podaj polecenie z właściwym manifestem.
4. Co zrobisz, jeśli monitoring pokaże duże RMSE, ale kandydat nie przejdzie bramki?
5. Jakie elementy trzeba dodać przed pracą zespołową i użyciem realnych danych?

Zapisz notebook. Pokaż prowadzącemu ostatni batch, oba manifesty oraz dowód rollbacku.

## 12. Zamknij warsztat i rozwiąż typowe problemy

1. Zapisz notebook przez **Ctrl+S**. Możesz pobrać jego kopię przez kliknięcie prawym
   przyciskiem pliku po lewej i **Download**.
2. Kliknij kartę terminala MLflow i naciśnij **Ctrl+C**. Poczekaj na zakończenie procesu.
3. W głównym terminalu uruchamiającym JupyterLab naciśnij **Ctrl+C**; potwierdź `y`,
   jeżeli Jupyter zapyta o zamknięcie. Zachowaj katalog wykonania jako dowód ćwiczenia.
4. Nowy pełny start: **Kernel → Restart Kernel and Clear Outputs of All Cells…**,
   potwierdź i wykonuj komórki od sekcji 0. Powstanie nowy katalog. Zatrzymaj stary
   serwer MLflow i uruchom polecenie wypisane dla nowego katalogu.

| Objaw | Co zrobić |
| --- | --- |
| Strona 8888 nie działa | Sprawdź terminal `uv run demo lab`, poczekaj na start, skopiuj adres z logu. |
| Port 8888 lub 5001 jest zajęty | Nie uruchamiaj drugiej kopii. Wróć do swojego poprzedniego terminala i zatrzymaj go Ctrl+C; procesów innych osób nie zamykaj. |
| `ModuleNotFoundError` | Sprawdź ścieżkę `Python:` do `.venv`, instalację `uv sync` i wybór kernela. |
| `NameError: catalog` / `WORKSPACE` | Wykonaj sekcje 0 i 1; pominięto poprzednią komórkę albo zrestartowano kernel. |
| Brak eksperymentu w MLflow | Sprawdź port 5001, wykonanie sekcji 3 i ścieżkę bazy w komendzie. Port 5000 pokazuje inne demo. |
| Brak rejestru / metryk | Wybierz Model training i powiększ okno. Do wersji służy Model registry, do testu parent run. |
| `Close the active MLflow run` | W nowej komórce wykonaj `mlflow.end_run()` i ponów właściwą komórkę; nie otwieraj zewnętrznego runu wokół sesji Kedro. |
| `database is locked` | Poczekaj na koniec zapisu; uruchamiaj jeden trening/scoring naraz. |
| Czerwony traceback | Przeczytaj ostatnią linię. Przy oczekiwanym błędzie widzisz `EXPECTED…`; pozostałych błędów nie pomijaj. |
| UI pokazuje stary champion | Odśwież stronę i sprawdź, czy oglądasz bazę bieżącego `WORKSPACE`. |
| Bramka nie przechodzi | Zatrzymaj wydanie, sprawdź raport i dane. Nie usuwaj asercji ani nie zmieniaj progów po zobaczeniu testu. |

## Źródła i dalszy rozwój

[Architektura projektu](../../docs/architecture.md),
[Kedro i Jupyter](https://docs.kedro.org/en/stable/integrations-and-plugins/notebooks_and_ipython/kedro_and_notebooks/),
[MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/workflow/).
Przykłady używają wersji z `uv.lock`, a nie dowolnej najnowszej instalacji.
Monitoring, promocja i batch są ręczne. Produkcja wymaga rzeczywistych etykiet,
walidacji czasowej, uprawnień, orkiestracji, migracji, kopii zapasowych i polityki retencji.
