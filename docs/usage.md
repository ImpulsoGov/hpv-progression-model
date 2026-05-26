# Usage guide

This guide covers installation, running the example notebook, and working directly with the Python API to build and evaluate custom simulations.

---

## Installation

### Option A — PDM (recommended for research and development)

[PDM](https://pdm-project.org) manages the virtual environment and locks dependency versions, making results reproducible.

```bash
# 1. Install PDM (once)
curl -sSL https://pdm-project.org/install.sh | bash

# 2. Clone the repository
git clone https://github.com/ImpulsoGov/hpv-progression-model.git
cd hpv-progression-model

# 3. Install the package and all dependencies into a managed venv
pdm install
```

All subsequent commands in this guide that start with `pdm run` execute inside that managed virtual environment.

### Option B — pip

```bash
pip install git+https://github.com/ImpulsoGov/hpv-progression-model.git
```

---

## Running the example notebook

The main analysis — a cost-effectiveness comparison of a messaging intervention to improve cervical cancer screening attendance in Brazil — lives in `examples/messaging_intervention.py`. It is a [Marimo](https://marimo.io) reactive notebook.

### Interactive (editable) mode

```bash
pdm run marimo edit examples/messaging_intervention.py
```

Marimo opens in your default browser. You can modify cell inputs (cohort size, compliance rate, intervention duration, etc.) and the results cells recompute automatically.

### Read-only preview (no install required)

Download `examples/messaging_intervention.html` from the repository and open it in any modern browser (Chrome, Firefox, Safari). The notebook renders as a static page.

> **Performance note:** A full production run (~83 700 individuals followed until death with a 48-month intervention window) takes **more than 24 hours** on a typical workstation. Reduce `sample_multiplier` in the notebook to get faster results at the cost of higher stochastic noise.

---

## Key parameters in the example notebook

| Variable | Default | Meaning |
|---|---|---|
| `screening_compliance` | 0.701 | Fraction of women who complete a screening within the recommended interval (from Brazil's 2019 DHS) |
| `screening_followup_loss` | 0.0329 | Fraction of positive screens lost to follow-up (not receiving treatment) |
| `intervention_duration_months` | 48 | Length of the messaging intervention window |
| `intervention_rate_ratio` | 3.1 / 1.3 ≈ 2.38 | Rate ratio for screening uptake in the intervention arm vs. control (from RCT pilot) |
| `quadrivalent_coverage` | 0.0 | Fraction of the cohort vaccinated (set > 0 to model vaccination) |
| `discount_rate` | 0.043 | Annual discount rate applied to future outcomes |
| `sample_multiplier` | 7 | Multiplies cohort size; reduce to speed up exploratory runs |

---

## Python API walkthrough

### 1. Building a cohort

`Cohort` is the top-level simulation object. It holds a set of `Individual` instances and advances them month by month.

```python
from hpv_progression_model.model import Cohort
from hpv_progression_model.params import INCIDENCES, PAP_SMEAR_3YRS_25_64

cohort = Cohort(
    age=25,                              # Starting age in years
    num_individuals=5_000,               # Cohort size
    incidences=INCIDENCES,               # Age-adjusted incidence dict (from params)
    screening_regimen=PAP_SMEAR_3YRS_25_64,  # Built-in Brazilian guideline
    screening_compliance=0.70,           # 70% compliance at the recommended interval
    screening_followup_loss=0.03,        # 3% of positive screens lost to follow-up
    vaccination_coverage=0.0,            # No vaccination in this example
)
```

The cohort is created with all individuals at `age=25`. If you want a realistic prevalence at that age, first warm it up from the age of sexual initiation (see below).

### 2. Warm-up: initialising age-group prevalence

In real analyses, the cohort is initialised at the average age of sexual initiation (~15 years) and then advanced to the study start age to build up a realistic HPV prevalence distribution:

```python
from hpv_progression_model.params import DEFAULT_AGE_FIRST_EXPOSURE

# Create cohort at first exposure
cohort = Cohort(age=DEFAULT_AGE_FIRST_EXPOSURE, num_individuals=5_000, ...)

# Advance month-by-month until age 25 (= 10 years × 12 months)
for _ in range(10 * 12):
    cohort.next()
```

The `Simulation` class wraps this pattern automatically via its `warm_up()` method.

### 3. Advancing the simulation

```python
# Advance by one month (all individuals progress, exposures occur, screenings are offered)
cohort.next()

# Inspect prevalences at the current time step
print(cohort.prevalences)

# Inspect accumulated outcomes (counts since simulation start)
print(cohort.outcomes_accumulated)
```

### 4. Using the Simulation runner

`Simulation` handles warm-up, copying the cohort for baseline/intervention arms, and running for a fixed interval:

```python
from hpv_progression_model.evaluation import Simulation

sim = Simulation(cohort, interval=120)   # 10-year follow-up
sim.run()

results = sim.results
print(results.summary)   # Pretty-printed outcome table
```

### 5. Comparing an intervention to a baseline

Use `evaluate_intervention()` to compare two cohort configurations side by side:

```python
from hpv_progression_model.evaluation import evaluate_intervention, apply_treatment
from hpv_progression_model.params import PAP_SMEAR_3YRS_25_64

# Define a higher-compliance screening regimen for the intervention arm
def intervention_rule(age, last_result):
    method, interval = PAP_SMEAR_3YRS_25_64.rule(age, last_result)
    return method, interval

# evaluate_intervention runs baseline and intervention in sequence and returns
# a dict of DichotomousComparison objects, one per ObservableOutcome
comparisons = evaluate_intervention(
    cohort=cohort,
    interval=120,
    treatment=lambda c: setattr(c, "_screening_compliance", 0.90),
)

for outcome, comparison in comparisons.items():
    print(f"{outcome}: RR={comparison.risk_ratio:.3f}, NNT={comparison.number_needed_to_treat}")
```

For the messaging-intervention analysis specifically, `apply_treatment` is used as a context manager to temporarily modify cohort parameters during the intervention window:

```python
from hpv_progression_model.evaluation import apply_treatment

with apply_treatment(cohort, screening_compliance=0.90):
    for _ in range(48):   # 48-month intervention window
        cohort.next()
# After the context exits, original parameters are restored
```

### 6. Computing time-discounted differences

```python
from hpv_progression_model.evaluation import compare_differences_in_outcomes

differences = compare_differences_in_outcomes(
    baseline_cohort_snapshot,
    intervention_cohort_snapshot,
    discount_rate=0.043,
)

# differences[ObservableOutcome.YLL] gives the discounted YLL difference
```

---

## Defining a custom screening regimen

A `ScreeningRegimen` wraps a callable that maps `(age, last_screening_result)` → `(ScreeningMethod, interval_months)`:

```python
from hpv_progression_model.common import ScreeningMethod, ScreeningRegimen, HPVInfectionState

# Custom method: higher sensitivity for CIN3+
hpv_dna_test = ScreeningMethod(
    sensitivity={
        HPVInfectionState.CIN2: 0.70,
        HPVInfectionState.CIN3: 0.90,
        HPVInfectionState.LOCAL_UNDETECTED: 0.95,
        HPVInfectionState.REGIONAL_UNDETECTED: 0.95,
        HPVInfectionState.DISTANT_UNDETECTED: 0.95,
    },
    specificity=0.92,
)

def my_rule(age, last_result):
    if last_result:           # Positive screen: retest sooner
        return hpv_dna_test, 12
    return hpv_dna_test, 60   # Negative: screen every 5 years

regimen = ScreeningRegimen(
    rule=my_rule,
    start_age=25,
    name="HPV DNA test every 5 years, 25+",
)
```

Pass `screening_regimen=regimen` when creating the `Cohort`.

---

## Reproducibility

The global random number generator (`RNG`) is seeded with `42` at import time. To run a different stochastic realisation:

```python
import numpy as np
from hpv_progression_model import common

common.RNG = np.random.default_rng(seed=123)
```

Set the seed before constructing any `Cohort` or `Individual` objects.

---

## CLI

The package provides a minimal command-line entry point:

```bash
hpv-model --version
hpv-model --debug-info   # prints Python version, OS, and dependency versions
```

All substantive analysis is done through the Python API or the Marimo notebook.

---

## Running the tests

```bash
pdm run pytest
```

Test configuration lives in [`config/pytest.ini`](../config/pytest.ini). Coverage reports are written to `htmlcov/`.
