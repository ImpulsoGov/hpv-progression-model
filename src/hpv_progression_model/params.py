# -*- coding: utf-8 -*-

"""Parameter definitions for the HPV progression model.

This module defines parameters related to the natural history of HPV infections, 
the effectiveness of treatments and vaccinations, and methods to convert survival 
curves and generate transition matrices for the simulation. These parameters are 
loaded from external YAML files and are used throughout the package to model 
HPV progression and interventions.

Attributes:
    TRANSITION_PROBABILITIES (dict[HPVGenotype, NDArray]): A dictionary mapping HPV genotypes to transition matrices that define the probabilities of progression between various health states in the model.
    QUADRIVALENT_EFFECTIVENESS (dict[HPVGenotype, float]): Effectiveness of the quadrivalent HPV vaccine for specific genotypes (e.g., HPV 16 and HPV
    18).
    SEE_AND_TREAT_EFFECTIVENESS (float): The effectiveness of the see-and-treat approach in preventing the progression of CIN 2 and CIN 3
    lesions.
    DEFAULT_DISCOUNT_RATE (float): The default yearly discount rate for the discounting benefits in the future.
    PAP_SMEAR (ScreeningMethod): Sensitivity and specificity values for Pap smear screening.
    PAP_SMEAR_3YRS_25_64 (ScreeningRegimen): A regimen where individuals aged 25-64 undergo Pap smear screenings every three years.

Example:
    Example usage of the module-level attributes:

    Accessing the Pap smear screening regimen:

    ```py
    >>> from hpv_progression_model.params import PAP_SMEAR_3YRS_25_64
    >>> PAP_SMEAR_3YRS_25_64.name
    'Pap smears every 3 years, 25-64 years old'
    ```

    Checking the effectiveness of the quadrivalent vaccine:

    ```py
    >>> from hpv_progression_model.params import QUADRIVALENT_EFFECTIVENESS
    >>> QUADRIVALENT_EFFECTIVENESS[HPVGenotype.HPV_16]
    0.7
    ```

Todo:
    * Implement the function to create a transition matrix from raw parameters.
    * Finalize the effectiveness data for Pap smear sensitivity and specificity.
    * Verify and update the effectiveness of the quadrivalent vaccine for other genotypes.
"""

import csv
import sys
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from os import PathLike
from pathlib import Path

import yaml
import numpy as np
from numpy.typing import NDArray

from .common import (
    CervicalCancerFIGOStage,
    FIGO_TO_LRD_STAGES,
    HPVGenotype,
    HPVInfectionState,
    ScreeningMethod,
    ScreeningRegimen,
)

# Directory containing seed files for simulation parameters
_SEEDS_DIR = Path(__file__).parents[2] / "seed"
_MORTALITY_FILE = _SEEDS_DIR / "gbd_mortality.csvy"
_NATURAL_HISTORY_PARAMS_FILE = _SEEDS_DIR / "natural_history_params.yaml"
_PREVALENCES_FILE = _SEEDS_DIR / "prevalences.yaml"
_REFERENCE_LIFE_TABLE_FILE = _SEEDS_DIR / "reference_life_table.csvy"
_SURVIVAL_CURVES_FILE = _SEEDS_DIR / "survival_curves.yaml"

MAX_FOLLOW_UP_DURATION = 100 * 12  # 100 years
CONSIDER_DETECTED_CANCERS_CURED_AT = 60  # cancers are deemed cured after 5 yrs

# Utility functions

def _annual_to_monthly_mortality_rate(annual_rate: float) -> float:
    """Converts annual mortality rates to monthly mortality rates.

    Args:
        annual_rate (float): The annual mortality rate.
    
    Returns:
        float: The monthly mortality rate.
    """
    monthly_rate = 1 - (1 - annual_rate) ** (1/12)
    return monthly_rate

def _convert_param_dict_to_array(
    param_dict: dict[
        tuple[HPVInfectionState, HPVInfectionState],
        dict[int, float],
    ],
) -> NDArray:
    """Converts a dictionary of natural history parameters into an array.

    Args:
        param_dict (dict): Dictionary containing natural history parameters.

    Returns:
        NDArray: A 3-dimensional NumPy array representing the monthly transition probabilities between different health states, where the first dimension corresponds to the time spent in the current health state, the second dimension corresponds to the health state transitioned from, and the third dimension corresponds to the health state transitioned to.
    """
    param_array = np.zeros((
        MAX_FOLLOW_UP_DURATION,
        len(HPVInfectionState),
        len(HPVInfectionState),
    ))
    for (state_source, state_destination), params in param_dict.items():
        t = 0
        value = params[0]
        while t < MAX_FOLLOW_UP_DURATION:
            value = params.get(t, value)
            param_array[
                t,
                state_source,
                state_destination,
            ] = value
            t += 1
    
    # Attribute the remaining probability to staying in the same state
    for t in range(MAX_FOLLOW_UP_DURATION):
        for state in HPVInfectionState:
            probabilities_sum = np.sum(param_array[t, state, :])
            param_array[t, state, state] += 1 - probabilities_sum

    return param_array


def _convert_survival_curves_figo_to_lrd(
    survival_curves: dict[CervicalCancerFIGOStage, dict[int, float]],
) -> dict[CervicalCancerFIGOStage, dict[int, float]]:
    """Converts survival curves from FIGO stages to LRD stages.

    This function aggregates the survival curves from FIGO cancer stages and
    reassigns them to their corresponding LRD (Local, Regional, Distant)
    stages based on the mapping in FIGO_TO_LRD_STAGES.

    Args:
        survival_curves (dict[CervicalCancerFIGOStage, dict[int, float]]): A
        dictionary containing survival curves for each FIGO stage.

    Returns:
        dict[CervicalCancerFIGOStage, dict[int, float]]: A dictionary with survival curves mapped to local, regional or distant stages.
    """
    # Internal data structures for counting and summing survival probabilities
    # TODO: consider implementing weighted averages or other methods for
    # combining survival curves from different FIGO stages.
    survival_curves_counts = {
        stage: defaultdict(int) for stage in set(FIGO_TO_LRD_STAGES.values())
    }
    survival_curves_sum = {
        stage: defaultdict(float) for stage in set(FIGO_TO_LRD_STAGES.values())
    }

    # Aggregate survival curves from FIGO stages into LRD stages
    for stage, survival_curve in survival_curves.items():
        for months_since_detection, p_survival in survival_curve.items():
            lrd_stage = FIGO_TO_LRD_STAGES[CervicalCancerFIGOStage[stage]]
            survival_curves_counts[lrd_stage][months_since_detection] += 1
            survival_curves_sum[lrd_stage][months_since_detection] += (
                p_survival
            )

    # Calculate the average survival curve for each LRD stage
    survival_curves_avg = {}
    for lrd_stage, survival_curve in survival_curves_sum.items():
        for months_since_detection, p_survival_sum in survival_curve.items():
            p_survival_count = (
                survival_curves_counts[lrd_stage][months_since_detection]
            )
            survival_curves_avg[lrd_stage] = {
                months_since_detection: p_survival_sum / p_survival_count
            }
    return survival_curves_avg


def _get_mortality_rates_from_survival_curve(
    survival_curve: dict[int, float],
) -> dict[int, float]:
    """Converts survival probabilities into monthly mortality rates.

    Args:
        survival_curve (dict[int, float]): A dictionary where keys are the number of months since cancer detection and values are the probability
        of survival.

    Returns:
        dict[int, float]: A dictionary where keys are months since detection and 
            values are the corresponding monthly mortality rates.
    
    Example:
        ```py
        >>> from hpv_progression_model.params import _get_mortality_rates_from_survival_curve
        >>> _get_mortality_rates_from_survival_curve({0: 1.0, 12: 0.98, 24: 0.953, 36: 0.937, 48: 0.901, 60: 0.856})
        {0: 0.0017, 12: 0.0023, 24: 0.0014, 36: 0.0033, 48: 0.0042}
        ```
    """
    mortality_rates: dict[int, float] = {}
    months_since_detection_last = 0
    p_survival_last = 1.0

    # Iterate through the survival curve and calculate mortality rates
    for months_since_detection, p_survival in survival_curve.items():
        mortality = p_survival / p_survival_last
        months_passed = months_since_detection - months_since_detection_last
        mortality_rate = 1 - mortality ** (1 / months_passed)
        mortality_rates[months_since_detection_last] = mortality_rate

        # Update for the next iteration
        months_since_detection_last = months_since_detection
        p_survival_last = p_survival

    return mortality_rates


def _process_gbd_age_groups(age_group: str) -> int:
    """Processes age groups from the GBD mortality rates.

    Args:
        age_group (str): A string representing an age group, e.g., "<5 years",
        "5-9 years", "10-14 years", "95+ years" etc.
    
    Returns:
        int: The start age of the age group.
    """
    age_group = (
        age_group.replace("<", "0-").replace("+", "").removesuffix(" years")
    )
    return int(age_group.split("-")[0])


def _process_state_pairs(
    state_pair_str: str,
) -> tuple[HPVInfectionState, HPVInfectionState]:
    """Processes comma-separated pairs of source and destination health states.

    Args:
        state_pair_str (str): A string representing a pair of health states, e.g., "INFECTED,HEALTHY".
    
    Returns:
        tuple[HPVInfectionState, HPVInfectionState]: A tuple containing the source and destination health states, in that order.
    
    Example:
        ```py
        >>> from hpv_progression_model.params import _process_state_pairs
        >>> _process_state_pairs("INFECTED,HEALTHY")
        (<HPVInfectionState.INFECTED: 0>, <HPVInfectionState.HEALTHY: 1>)
        ```
    """
    state_source, state_destination = state_pair_str.split(",", maxsplit=1)
    return (
        HPVInfectionState[state_source.upper()],
        HPVInfectionState[state_destination.upper()],
    )


def _process_ranges(range_str: str) -> float:
    """Processes a comma-separated range string into a single float value.

    Args:
        range_str (str): A string representing a range of values, e.g.,
        "0.001,0.005".
    
    Returns:
        float: The average of the range values.
    """
    # TODO: Use squigglepy to return a distribution instead of the simple
    # average of the range values.
    min_value, max_value = range_str.split(",")
    return (float(min_value) + float(max_value)) / 2


def _repeat_parameters_for_all_types(
    param_dict: dict[str, dict[int, float | str]],
) -> dict:
    """Repeats the same parameters for all available HPV genotypes.
    
    Args:
        param_dict (dict): A dictionary containing the parameters to be repeated.

    Returns:
        dict: A dictionary containing the repeated parameters.
    """
    repeated_params = {}
    for genotype in HPVGenotype:
        repeated_params[genotype.name] = param_dict
    return repeated_params


def load_epidemiological_parameters(
    natural_history_params_file: PathLike,
    survival_curves_file: PathLike,
) -> dict[
    HPVGenotype,
    dict[tuple[HPVInfectionState, HPVInfectionState], float],
]:
    """Loads parameters and survival curves from YAML files""" 
    with open(natural_history_params_file, "r") as f:
        natural_history_params = yaml.load(f, Loader=yaml.FullLoader)

    with open(survival_curves_file, "r") as f:
        survival_curves_figo = yaml.load(f, Loader=yaml.FullLoader)

    # Repeat genotype-agnostic parameters for all available HPV genotypes
    type_agnostic_params = natural_history_params.pop("ALL_TYPES")
    for k, v in type_agnostic_params.items():
        for genotype, params in _repeat_parameters_for_all_types(
            {k: v},
        ).items():
            natural_history_params[genotype].update(params)

    # process "SOURCE,DESTINATION" stage pair strings into tuples
    for genotype in natural_history_params:
        state_pairs = deepcopy(list(natural_history_params[genotype]))
        for state_pair in state_pairs:
            value = natural_history_params[genotype].pop(state_pair)
            natural_history_params[genotype][
                _process_state_pairs(state_pair)
            ] = value

    # read values that are expressed as ranges and convert them to averages
    for genotype in natural_history_params:
        for state_pair, params in natural_history_params[genotype].items():
            for period, value in params.items():
                natural_history_params[genotype][state_pair][period] = (
                    _process_ranges(value) if isinstance(value, str) else value
                )

    # Compute mortality rates for each LRD stage based on survival curves
    # and assign them to the natural history parameters
    survival_curves_lrd = _convert_survival_curves_figo_to_lrd(
        survival_curves_figo,
    )
    mortality_rates = {
        stage: _get_mortality_rates_from_survival_curve(survival_curve)
        for stage, survival_curve in survival_curves_lrd.items()
    }
    for genotype, genotype_params in natural_history_params.items():
        for stage_at_detection, mortality_rate in mortality_rates.items():
            # strip mortality after the cancer is considered cured
            mortality_rate[CONSIDER_DETECTED_CANCERS_CURED_AT] = 0.0
            mortality_rate = {
                k: v for k, v in mortality_rate.items()
                if k <= CONSIDER_DETECTED_CANCERS_CURED_AT
            }
            genotype_params[(
                stage_at_detection,
                HPVInfectionState.DECEASED,
            )] = mortality_rate

    # convert genotype strings into HPVGenotype enums
    for genotype in HPVGenotype:
        if genotype.name not in natural_history_params:
            raise ValueError(
                f"No natural history parameters found for {genotype.value}",
            )
        natural_history_params[genotype] = (
            natural_history_params.pop(genotype.name)
        )

    return natural_history_params


def load_mortality_rates(
    mortality_file: PathLike,
    filter_locations: list[str] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Loads mortality rates from a CSV file.

    Read a CSV file containing the all-cause and cervical cancer-specific
    mortality rates.

    Args:
        mortality_file (PathLike): The path to the CSV file containing the
        mortality rates.
    
    Returns:
        dict[int, float]: A dictionary mapping ages to mortality rates.
    """
    mortality_rates_all_causes = {}
    mortality_rates_cervical_cancer = {}

    # Open the CSV file and read the data
    with open(mortality_file, mode="r", encoding="utf-8") as file:
        
        reader = csv.reader(file)
        for row in reader:
            if row and not row[0].startswith("#"): # Skip metadata comments
                if row[0] == "measure_id":  # Skip the header
                    field_names = list(row)
                # Append the age and life expectancy to the lists
                row_data ={
                    field_names[i]: row[i]
                    for i in range(len(field_names))
                }
                if row_data["metric_name"] != "Rate":
                    continue
                if (
                    filter_locations
                    and row_data["location_name"] not in filter_locations
                ):
                    continue

                age_start = _process_gbd_age_groups(row_data["age_name"])
                mortality_rate = float(row_data["val"]) / 100000
                
                if row_data["cause_name"] == "Cervical cancer":
                    mortality_rates_cervical_cancer[age_start] = (
                        _annual_to_monthly_mortality_rate(mortality_rate)
                    )
                if row_data["cause_name"] == "All causes":
                    mortality_rates_all_causes[age_start] = (
                        _annual_to_monthly_mortality_rate(mortality_rate)
                    )

    return mortality_rates_cervical_cancer, mortality_rates_all_causes


def load_prevalences(prevalences_file: PathLike) -> dict[HPVGenotype, float]:
    """Loads prevalences from a YAML file.
    
    Read a YAML file containing prevalences of HPV genotypes in a given
    population, and return a dictionary mapping HPV genotypes to prevalences.

    Args:
        prevalences_file (PathLike): The path to the YAML file containing the
        prevalences.
    
    Returns:
        dict[HPVGenotype, float]: A dictionary mapping HPV genotypes to their
        prevalences.
    """
    with open(prevalences_file, "r") as f:
        prevalences = yaml.load(f, Loader=yaml.FullLoader)
    
    # convert genotype strings into HPVGenotype enums
    for genotype in HPVGenotype:
        if genotype.name not in prevalences:
            raise ValueError(
                f"No prevalence found for {genotype.value}",
            )
        prevalences[genotype] = prevalences.pop(genotype.name)
    return prevalences


def load_reference_life_table(
    reference_life_table_file: PathLike,
) -> dict[int, float]:
    """Loads a reference life table from a CSV file.
    
    Read a CSV file containing a reference life table obtained from the Global Burden of Disease Study, and return a dictionary mapping ages to life expectancies. 

    Args:
        reference_life_table_file (PathLike): The path to the CSV file containing the reference life table.
    
    Returns:
        dict[int, float]: A dictionary mapping ages to life expectancies.
    """
    reference_life_table = {}

    # Open the CSV file and read the data
    with open(reference_life_table_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        
        # Skip lines starting with '#'
        for row in reader:
            if row and not row[0].startswith("#"):
                if row[0] == "Age":  # Skip the header
                    continue
                # Append the age and life expectancy to the lists
                age, life_expectancy = row[0], row[1]
                reference_life_table[int(age)] = float(life_expectancy)

    return reference_life_table


def estimate_life_expectancy(
    age: int,
    reference_life_table: dict[int, float],
) -> float:

    # Check if the input age is greater than the maximum age in the table
    max_age = max(reference_life_table.keys())
    if age > max_age:
        return reference_life_table[max_age]
    
    # If the age is exactly in the table, return the corresponding life expectancy
    try:
        return reference_life_table[age]
    
    # Otherwise, interpolate it
    except KeyError:
        return np.interp(
            age,
            list(reference_life_table.keys()),
            list(reference_life_table.values()),
        ).item()


def estimate_incidence(
    prevalence: float,
    clearance_rates: Iterable[float],
    followup_duration: int = MAX_FOLLOW_UP_DURATION,
) -> float:  
    """
    Estimate the average incidence rate given prevalence and clearance rate.
    
    Args:
        prevalence (float): The disease prevalence in the population.
        clearance_rates (NDArray): The monthly probabilities of regression to
        the healthy state.
        followup_duration (int): The maximum number of months to simulate for the cohort.

    Returns:
        float: Estimated incidence rate.

    Notes:
        This is a naive estimation that takes into account only the disease
        _regression_ to the healthy state, ignoring its progression to more
        severe states, which can prolong the disease duration.
    """

    # Time vector from 0 to max_time months
    weighted_disease_durations = []
    prop_infected = np.arange(followup_duration + 1)
    prop_infected[0] = 1.0
    clearance_rate = clearance_rates[0]
    
    # Probability of remaining infected after t months
    for t in range(0, followup_duration + 1):
        clearance_rate = clearance_rates.get(t, clearance_rate)
        cleared = (prop_infected[t-1] * clearance_rate)
        weighted_disease_durations.append(t * cleared)
        prop_infected[t] = prop_infected[t-1] - cleared
    
    avg_disease_duration = np.sum(weighted_disease_durations)
    
    # Use prevalence and average disease duration to estimate incidence rate
    incidence_rate = prevalence / avg_disease_duration
    
    return incidence_rate

# Default yearly discount rate for the discounting benefits in the future
DEFAULT_DISCOUNT_RATE: float = 0.043

# Average age of the first sexual intercourse
# Source: Associação Hospitalar Moinhos de Vento. (2020). Estudo
# epidemiológico sobre a prevalência nacional de infecção pelo HPV
# (POP-BRASIL) - 2015-2017 (1st ed., p.20). Porto Alegre, RS: Associação
# Hospitalar Moinhos de Vento.
# https://drive.google.com/file/d/1joUA-2nkBiv5sABB2UOfQMRczuH1j5Cs/view
DEFAULT_AGE_FIRST_EXPOSURE: float = 15.2

# Transition probabilities for each HPV genotype
_epidemiological_params = load_epidemiological_parameters(
    natural_history_params_file=_NATURAL_HISTORY_PARAMS_FILE,
    survival_curves_file=_SURVIVAL_CURVES_FILE,
)
TRANSITION_PROBABILITIES: dict[HPVGenotype, NDArray] = {
    genotype: _convert_param_dict_to_array(params)
    for genotype, params in _epidemiological_params.items()
}

# Mortality rates for all other causes not related to cervical cancer
(
    _cervical_cancer_mortality_rates,
    _all_cause_mortality_rates,
) = load_mortality_rates(
    mortality_file=_MORTALITY_FILE,
    filter_locations=["Brazil"],
)
NON_CERVICAL_CANCER_MORTALITY_RATES: dict[int, float] = {
   t: _all_cause_mortality_rates[t]
   - _cervical_cancer_mortality_rates.get(t, 0.0)
   for t in _all_cause_mortality_rates
}

_prevalences = load_prevalences(_PREVALENCES_FILE)
INCIDENCES = {
    genotype: estimate_incidence(
        prevalence,
        clearance_rates=_epidemiological_params[genotype][(
            HPVInfectionState.INFECTED,
            HPVInfectionState.HEALTHY,
        )],
    )
    for genotype, prevalence in _prevalences.items()
}

# Probability that an individual becomes immune to a specific HPV genotype
# after an infection
# Source: Kim J. J., Burger E. A., Regan C. & Sy S. (2017). Screening for cervical cancer in primary care: a decision analysis for the US Preventive Services Task Force. JAMA. https://doi.org/10.1001/jama.2017.19872
NATURAL_IMMUNITY: float = 0.5  

# Effectiveness of the quadrivalent vaccine for specific HPV genotypes
# Source: Arbyn, M., Xu, L., Simoens, C., & Martin-Hirsch, P. P. L. (2018).
# Prophylactic vaccination against human papillomaviruses to prevent cervical cancer and its precursors. Cochrane Database of Systematic Reviews, 2018(5).
# https://doi.org/10.1002/14651858.CD009069.pub3
# Summary of findings 3. ("HPV vaccine effects in adolescent girls and women
# regardless of HPV DNA status at baseline"), Outcome "CIN2+ associated with
# HPV16/18", 15 to 26 years
QUADRIVALENT_EFFECTIVENESS: dict[HPVGenotype, float] = {
    HPVGenotype.HPV_16: 1 - 0.46,
    HPVGenotype.HPV_18: 1 - 0.46,
}

# Effectiveness of the "See and Treat" approach
# Source: Arbyn et al., cited by INCA (2016), section "Seguimento
# pós-tratamento de NIC II/III", p. 82 (available from: 
# <https://www.inca.gov.br/publicacoes/livros/
# diretrizes-brasileiras-para-o-rastreamento-do-cancer-do-colo-do-utero>).
# NOTE: We could not locate the number cited by INCA (2016) in the study mentioned in their references.
SEE_AND_TREAT_EFFECTIVENESS: float = 0.92

NO_SCREENING = ScreeningMethod(sensitivity={}, specificity=1.0)

# Definition of Pap smear sensitivity and specificity
PAP_SMEAR = ScreeningMethod(
# Source: Coppleson, L. W., & Brown, B. (1974). Estimation of the screening
# error rate from the observed detection rates in repeated cervical cytology.
# American Journal of Obstetrics and Gynecology, 119(7), 953-958.
# https://doi.org/10.1016/0002-9378(74)90013-1
# Sensitivities are based on Coppleson & Brown (1974), Tables I and II. We
# considered "dysplasia" to mean a CIN2 lesion; and "CIS" (carcinoma in situ)
# to mean a CIN2 lesion. This is in line with INCA (2016) equivalence between
# various nomenclatures for citopathological and histological classifications
# of cervical lesions.
    sensitivity={
        HPVInfectionState.CIN2: 1-0.399,
        HPVInfectionState.CIN3: 1-0.272,
        HPVInfectionState.LOCAL_UNDETECTED: 1-0.24,
        HPVInfectionState.REGIONAL_UNDETECTED: 1-0.24,
        HPVInfectionState.DISTANT_UNDETECTED: 1-0.24,
    },
# Source: McCrory, D. C., Matchar, D. B., Bastian, L., Datta, S., Hasselblad,
# V., Hickey, J., ... & Nanda, K. (1999). Evaluation of cervical cytology.
# Evidence report/technology assessment (Summary), (5), 1-6.
# https://europepmc.org/article/MED/11925972/NBK32970
# For specificity, we assume the HSIL/CIN2-3 diagnosis is the threshold for
# demanding a colposcopy, following the guidelines prescribed by INCA (2016),
# "Quadro 4", p. 31 (available from: 
# <https://www.inca.gov.br/publicacoes/livros
# /diretrizes-brasileiras-para-o-rastreamento-do-cancer-do-colo-do-utero>).
# We then use the ROC curve from McCrory et al. (1999) Figure 17 to get the
# specificity associated to the CIN-2 sensitivity outlined above. We used the
# WebPlotDigitizer tool (available from: 
# https://web.eecs.utk.edu/~dcostine/personal/PowerDeviceLib/DigiTest
# /index.html) to extract the approximate data point.
    specificity=1-0.14,
)


def _pap_smear_3yrs_25_64_rule(
    age: int | float,
    last_screening_result: bool | None,
) -> tuple[ScreeningMethod, int]:
    if age < 25:
        return PAP_SMEAR, int(12 * (25 - age))
    if age >= 25 and age < 65:
        return PAP_SMEAR, 36
    return NO_SCREENING, sys.maxsize


# Screening regimen for Pap smears every 3 years for ages 25-64
PAP_SMEAR_3YRS_25_64 = ScreeningRegimen(
    name="Pap smears every 3 years, 25-64 years old",
    description=(
        "A screening regimen where all people aged 25-64 years old are "
        "prescribed conventional pap smears every 3 years, independently of "
        "their last screening result."
    ),
    rule=_pap_smear_3yrs_25_64_rule,
    start_age=25,
)

NATURAL_HISTORY = ScreeningRegimen(
    name="Natural history, with no screening",
    description=(
        "A placeholder screening regimen where all people follow the disease "
        "natural history."
    ),
    rule=lambda *args, **kwargs: (NO_SCREENING, sys.maxsize),
    start_age=0,
)

reference_life_table = load_reference_life_table(
    reference_life_table_file=_REFERENCE_LIFE_TABLE_FILE,
)
get_life_expectancy = partial(
    estimate_life_expectancy,
    reference_life_table=reference_life_table,
)
