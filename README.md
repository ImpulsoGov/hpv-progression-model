# HPV Progression Model

[![ci](https://github.com/ImpulsoGov/hpv-progression-model/workflows/ci/badge.svg)](https://github.com/ImpulsoGov/hpv-progression-model/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://ImpulsoGov.github.io/hpv-progression-model/)
[![pypi version](https://img.shields.io/pypi/v/hpv_progression_model.svg)](https://pypi.org/project/hpv_progression_model/)

A semi-markovian microsimulation model of the natural history of Human Papillomavirus (HPV) infection and its outcomes — precancerous cervical lesions and invasive cancers — with support for evaluating vaccination, screening, and treatment interventions. Originally developed for cost-effectiveness analysis (CEA) of cervical cancer prevention programmes in Brazil.

## Requirements

- Python 3.11+
- [PDM](https://pdm-project.org/) (recommended for development installs)

## Installation

### From PyPI / GitHub (end users)

```bash
pip install git+https://github.com/ImpulsoGov/hpv-progression-model.git
```

### From source (developers / researchers)

```bash
# Install PDM (package manager)
curl -sSL https://pdm-project.org/install.sh | bash

# Clone and install with all development dependencies
git clone https://github.com/ImpulsoGov/hpv-progression-model.git
cd hpv-progression-model
pdm install
```

## Quickstart

The primary way to interact with the model is through the example notebook.
To open it in an interactive [Marimo](https://marimo.io) environment:

```bash
pdm run marimo edit examples/messaging_intervention.py
```

> **Note:** A full simulation run (83 000+ individuals followed until death) can take **more than 24 hours** on a typical workstation. Start with a smaller cohort to explore the model.

For a read-only, no-install preview, open `examples/messaging_intervention.html` in any modern browser after downloading the file from the repository.

### Minimal Python example

```python
from hpv_progression_model.model import Cohort
from hpv_progression_model.params import (
    INCIDENCES,
    PAP_SMEAR_3YRS_25_64,
)
from hpv_progression_model.evaluation import Simulation, evaluate_intervention

# Build a small cohort of 1 000 women aged 25
cohort = Cohort(
    age=25,
    num_individuals=1_000,
    incidences=INCIDENCES,
    screening_regimen=PAP_SMEAR_3YRS_25_64,
    screening_compliance=0.70,
)

# Run for 10 years (120 months) and inspect accumulated outcomes
simulation = Simulation(cohort, interval=120)
simulation.run()
print(simulation.results.summary)
```

## Project structure

```
hpv-progression-model/
├── src/hpv_progression_model/   # Python package
│   ├── common.py                # Enumerations, base classes, shared constants
│   ├── params.py                # Parameter loading and pre-built regimens/constants
│   ├── model.py                 # Core simulation classes (HPVInfection, Individual, Cohort)
│   ├── evaluation.py            # Simulation runner and outcome comparison utilities
│   └── cli.py                   # Command-line entry point
├── seed/                        # Epidemiological input data (YAML / CSVY)
│   ├── natural_history_params.yaml   # State-transition probabilities by genotype
│   ├── disease_duration.yaml         # Mean infection durations
│   ├── prevalences.yaml              # HPV prevalence by genotype (Brazil)
│   ├── incidence_by_age.csvy         # Age-adjusted relative incidence curve
│   ├── gbd_mortality.csvy            # GBD 2021 all-cause mortality (Brazilian females)
│   ├── reference_life_table.csvy     # Life expectancy by age
│   └── survival_curves.yaml          # Cervical cancer survival by FIGO stage
├── examples/
│   └── messaging_intervention.py    # Main analysis notebook (Marimo)
├── docs/                        # Extended documentation
│   ├── model.md                 # Model theory, states, and data sources
│   └── usage.md                 # Detailed usage guide and API walkthrough
└── tests/                       # Unit tests
```

## Model overview

The model simulates the life of each individual woman as a **semi-Markov process** with eleven mutually exclusive health states:

```
HEALTHY → INFECTED → CIN2 / CIN3 → Local cancer → Regional cancer → Distant cancer → DECEASED
```

Each individual can progress, regress, or die each month according to **time-in-state– and genotype-specific transition probabilities**, estimated from published clinical literature. Three types of intervention can be applied to any cohort:

| Intervention | Effect |
|---|---|
| **Vaccination** (quadrivalent) | Reduces susceptibility to HPV 16/18 (and cross-protection for 31/45) |
| **Screening** (e.g. Pap smear) | Detects precancerous lesions and undetected cancers; triggers treatment |
| **Treatment** (see-and-treat) | Removes CIN2/CIN3 lesions, returning individuals to the healthy/infected state |

Outcomes tracked per time step include cytology exams, lesion and cancer detections, treatment procedures (colposcopies, excisions, biopsies), deaths, and Years of Life Lost (YLL) — both nominal and time-discounted.

See [docs/model.md](docs/model.md) for a detailed description of the model logic, transition diagram, and epidemiological parameters. Or check the AI-generated video explainer below:

<video src="https://github.com/ImpulsoGov/hpv-progression-model/releases/download/v0.1.0/HPV_Progression_Model.mp4" width="320" height="240" controls></video>

## Documentation

| Document | Contents |
|---|---|
| [Model description](docs/model.md) | States, transitions, semi-Markov formulation, data sources |
| [Usage guide](docs/usage.md) | Installation, running simulations, defining interventions, reading results |
| [API reference](https://ImpulsoGov.github.io/hpv-progression-model/reference/) | Full auto-generated reference for all public classes and functions |

## References

The model parameters were drawn from the following sources:

- Kim, J.J. et al. (2017). Optimal cervical cancer screening in women vaccinated against human papillomavirus. *JAMA Oncology*, 3(6), 809-816. [doi:10.1001/jama.2017.19872](https://doi.org/10.1001/jama.2017.19872) — state-transition probabilities
- Muñoz, N. et al. (2004). Epidemiologic classification of human papillomavirus types associated with cervical cancer. *New England Journal of Medicine*, 350(11), 1133-1140. [doi:10.1056/NEJMoa031444](https://doi.org/10.1056/NEJMoa031444) — age-specific incidence and infection duration
- Wendland, E.M. et al. (2020). Prevalence of HPV infection among sexually active adolescent girls and young women in Brazil. *Scientific Reports*, 10, 4920. [doi:10.1038/s41598-020-61582-2](https://doi.org/10.1038/s41598-020-61582-2) — HPV genotype prevalence in Brazil
- Bandeira, I.C.J. et al. (2024). Genotypic characterization of the HPV-female Brazilian population. *PLOS ONE*, 19(6), e0305122. [doi:10.1371/journal.pone.0305122](https://doi.org/10.1371/journal.pone.0305122) — additional genotype data
- Carmo, C.C. & Luiz, R.R. (2011). Survival of a cohort of women with cervical cancer diagnosed in a Brazilian cancer center. *Revista de Saúde Pública*, 45(4), 661-667. [doi:10.1590/S0034-89102011005000029](https://doi.org/10.1590/S0034-89102011005000029) — cervical cancer survival curves for Brazil
- Global Burden of Disease Study 2021 — all-cause mortality and life expectancy for Brazilian females. [GBD Results Tool](https://vizhub.healthdata.org/gbd-results)

## License

MIT — see [LICENSE](LICENSE).
