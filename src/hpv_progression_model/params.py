# -*- coding: utf-8 -*-

"""Parameter definitions for the HPV progression model.

This module defines parameters related to the natural history of HPV infections, the effectiveness of treatments and vaccinations, and methods to 
convert survival curves and generate transition matrices for the simulation. These parameters are loaded from external YAML files and are used throughout 
the package to model HPV progression and interventions.

Attributes:
    TRANSITION_PROBABILITIES (dict[HPVGenotype, NDArray]): A dictionary mapping HPV genotypes to transition matrices that define the probabilities of progression between various health states in the model.
    QUADRIVALENT_EFFECTIVENESS (dict[HPVGenotype, float]): Effectiveness of the quadrivalent HPV vaccine for specific genotypes (e.g., HPV 16 and HPV
    18).
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

# REFACTOR: This module needs urgent refactoring! It's a mess.

import csv
import sys
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import scipy.stats as stats
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
_DISEASE_DURATION_FILE = _SEEDS_DIR / "disease_duration.yaml"
_INCIDENCES_BY_AGE_FILE = _SEEDS_DIR / "incidence_by_age.csvy"
_MORTALITY_FILE = _SEEDS_DIR / "gbd_mortality.csvy"
_NATURAL_HISTORY_PARAMS_FILE = _SEEDS_DIR / "natural_history_params.yaml"
_PREVALENCES_FILE = _SEEDS_DIR / "prevalences.yaml"
_REFERENCE_LIFE_TABLE_FILE = _SEEDS_DIR / "reference_life_table.csvy"
_SURVIVAL_CURVES_FILE = _SEEDS_DIR / "survival_curves.yaml"

MAX_FOLLOW_UP_DURATION = 100 * 12  # 100 years
CONSIDER_DETECTED_CANCERS_CURED_AT = 60  # cancers are deemed cured after 5 yrs

# Average age of the first sexual intercourse
# Source: Associação Hospitalar Moinhos de Vento. (2020). Estudo
# epidemiológico sobre a prevalência nacional de infecção pelo HPV
# (POP-BRASIL) - 2015-2017 (1st ed., p.20). Porto Alegre, RS: Associação
# Hospitalar Moinhos de Vento.
# https://drive.google.com/file/d/1joUA-2nkBiv5sABB2UOfQMRczuH1j5Cs/view
DEFAULT_AGE_FIRST_EXPOSURE: int = 15


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


def _convert_figo_to_lrd(
    dicts_by_stage: dict[CervicalCancerFIGOStage, dict[Any, float]],
) -> dict[CervicalCancerFIGOStage, dict[Any, float]]:
    """Converts survival curves from FIGO stages to LRD stages.

    This function aggregates the survival curves from FIGO cancer stages and
    reassigns them to their corresponding LRD (Local, Regional, Distant)
    stages based on the mapping in FIGO_TO_LRD_STAGES.

    Args:
        dicts_by_stage (dict[CervicalCancerFIGOStage, dict[Any, float]]): A
        (FIGO stage-indexed) dictionary of dictionaries.

    Returns:
        dict[CervicalCancerFIGOStage, dict[Any, float]]: A dictionary mapping
        the values to local, regional or distant stages instead.
    """
    # Internal data structures for counting and summing survival probabilities
    # TODO: consider implementing weighted averages or other methods for
    # combining survival curves from different FIGO stages.
    values_count = {
        stage: defaultdict(int) for stage in set(FIGO_TO_LRD_STAGES.values())
    }
    values_sum = {
        stage: defaultdict(float) for stage in set(FIGO_TO_LRD_STAGES.values())
    }

    # Aggregate sum and count of values from FIGO stages into LRD stages
    for stage, values_dict in dicts_by_stage.items():
        for key, value in values_dict.items():
            lrd_stage = FIGO_TO_LRD_STAGES[CervicalCancerFIGOStage[stage]]
            values_count[lrd_stage][key] += 1
            values_sum[lrd_stage][key] += value

    # Calculate the average value for each LRD stage
    values_avg = {}
    for lrd_stage, values_dict in values_sum.items():
        values_avg[lrd_stage] = {}
        for key, sum_ in values_dict.items():
            count = values_count[lrd_stage][key]
            values_avg[lrd_stage].update({key: sum_ / count})
    return values_avg


def _get_normal_distribution_from_age_cutoff_and_mode(
    cutoff: int | float,
    number_below_cutoff: int,
    number_above_cutoff: int,
    mode: int | float,
) -> tuple[float, float]:
    """Estimate the probability of a normal distribution from a cutoff.

    Args:
        cutoff (int | float): The cutoff value.
        number_below_cutoff (int): The number of values below the cutoff.
        number_above_cutoff (int): The number of values above the cutoff.
        mode (int | float): The known or estimated mode of the distribution. Defaults to None, used to estimate the distribution `mu` parameter.


    Returns:
        float: The mu parameter of the normal distribution.
        float: The sigma parameter of the normal distribution.
    """
    prop_below_cuttoff = (
        number_below_cutoff / (number_above_cutoff + number_below_cutoff)
    )
    z = stats.norm.ppf(prop_below_cuttoff)
    mu = mode
    sigma = (cutoff - mu) / z
    return mu, sigma


def _get_normal_distribution_from_age_cutoff_and_stdev(
    cutoff: int | float,
    number_below_cutoff: int,
    number_above_cutoff: int,
    stdev: float,
) -> tuple[float, float]:
    """Estimate the probability of a normal distribution from a cutoff.

    Args:
        cutoff (int | float): The cutoff value.
        number_below_cutoff (int): The number of values below the cutoff.
        number_above_cutoff (int): The number of values above the cutoff.
        stdev (float): The known or estimated standard deviation of the sample, used as the distribution `sigma` parameter.

    Returns:
        float: The mu parameter of the normal distribution.
        float: The sigma parameter of the normal distribution.
    """
    prop_below_cuttoff = (
        number_below_cutoff / (number_above_cutoff + number_below_cutoff)
    )
    z = stats.norm.ppf(prop_below_cuttoff)
    mu = cutoff - z * stdev
    sigma = stdev
    return mu, sigma


def _estimate_additional_mortality_from_cancer(
    cancer_survival_curve: dict[int, float],
    non_cancer_mortality_rates: dict[int, float],
    sample_age_distribution: dict[int, float],
) -> dict[int, float]:
    additional_mortalities = {}
    cancer_survival_curve_months = list(sorted(cancer_survival_curve.keys()))
    for month_idx in range(len(cancer_survival_curve_months)-1):
        curve_month = cancer_survival_curve_months[month_idx]
        curve_month_next = cancer_survival_curve_months[month_idx + 1]
        months_passed = curve_month_next - curve_month
        age_group_weighted_additional_mortality_sum = 0.0
        for age, num_individuals in sample_age_distribution.items():
            non_cancer_mortality = 1 - (
                (1 - non_cancer_mortality_rates.get(age, 0.0)) ** months_passed
            )
            actual_mortality = 1 - (
                cancer_survival_curve[curve_month_next]
                / cancer_survival_curve[curve_month]
            )
            additional_mortality = actual_mortality - non_cancer_mortality
            age_group_weighted_additional_mortality_sum += (
                num_individuals * additional_mortality
            )
        # average out the age-weighted additional mortalities and store them
        additional_mortalities[curve_month] = (
            age_group_weighted_additional_mortality_sum
            / sum(sample_age_distribution.values())
        )
    return additional_mortalities


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
) -> dict[
    HPVGenotype,
    dict[tuple[HPVInfectionState, HPVInfectionState], float],
]:
    """Loads parameters and survival curves from YAML files""" 
    with open(natural_history_params_file, "r") as f:
        natural_history_params = yaml.load(f, Loader=yaml.FullLoader)

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


def load_cancer_additional_mortalities(
    survival_curves_file: PathLike,
    non_cancer_mortality_rates: dict[int, float],
    mode_cancer_prevalence: int | float,
    consider_detected_cancers_cured_at: int = 60,
) -> dict[tuple[HPVInfectionState, HPVInfectionState], float]:

    with open(survival_curves_file, "r") as f:
        survival_curves_contents = yaml.load(f, Loader=yaml.FullLoader)
        survival_curves_figo = survival_curves_contents[
            "curves_by_stage_at_detection"
        ]
        sample_by_stage_and_age = survival_curves_contents["sample_by_age"]
    
    additional_mortalities_by_figo_stage = {}

    age_cutoff = max(sample_by_stage_and_age["I"].keys())
    num_individuals_below_cutoff_total = sum(
        sample_by_stage_and_age[stage][0]
        for stage in sample_by_stage_and_age
    )
    num_individuals_above_cutoff_total = sum(
        sample_by_stage_and_age[stage][age_cutoff]
        for stage in sample_by_stage_and_age
    )
    num_individuals_in_sample = (
        num_individuals_below_cutoff_total + num_individuals_above_cutoff_total
    )
    # first, get the sample estimated standard deviation using the mode
    _, sigma = _get_normal_distribution_from_age_cutoff_and_mode(
        age_cutoff,
        num_individuals_below_cutoff_total,
        num_individuals_above_cutoff_total,
        mode_cancer_prevalence,
    )
    # then, get the estimated average ages for each stage subgroup, assuming
    # they all inherit the same standard deviation as the larger sample
    for stage_figo, sample_by_age in sample_by_stage_and_age.items():
        sample_age_distribution = {}
        mu, _ = _get_normal_distribution_from_age_cutoff_and_stdev(
            age_cutoff,
            sample_by_age[0],
            sample_by_age[age_cutoff],
            sigma,
        )

        # use the estimated distribution parameters to estimate the number of
        # individuals per 5-year age group in the sample
        for age in range(0, 105, 5):
            if age == 0:
                prob = stats.norm.cdf(age, mu, sigma)
            elif age == 100:
                prob = 1 - stats.norm.cdf(95, mu, sigma)
            else:
                prob = (
                    stats.norm.cdf(age, mu, sigma)
                    - stats.norm.cdf(age - 5, mu, sigma)
                )
            sample_age_distribution[age] = int(
                prob * num_individuals_in_sample
            )

        additional_mortalities_by_figo_stage[stage_figo] = (
            _estimate_additional_mortality_from_cancer(
                survival_curves_figo[stage_figo],
                non_cancer_mortality_rates,
                sample_age_distribution,
            )
        )
    additional_mortalities_lrd = _convert_figo_to_lrd(
        additional_mortalities_by_figo_stage,
    )
    
    # cap additional mortality to some months, after which the cancer is
    # considered cured (i.e., no additional mortality comes from it)
    additional_mortalities_capped = {}
    for stage in additional_mortalities_lrd.keys():
        additional_mortalities_capped[stage] = {
            k: v for k, v in additional_mortalities_lrd[stage].items()
            if k <= consider_detected_cancers_cured_at
        }
        additional_mortalities_capped[stage][
            consider_detected_cancers_cured_at
        ] = 0.0

    undetected_cancer_additional_mortalities = {}
    for detected_stage, detected_mortality in additional_mortalities_capped.items():
        undetected_mortality = max(detected_mortality.values())
        undetected_stage = HPVInfectionState(detected_stage - 1)
        undetected_cancer_additional_mortalities[undetected_stage] = {
            0: undetected_mortality
        }

    additional_mortalities_capped.update(
        undetected_cancer_additional_mortalities,
    )

    return {
        (stage, HPVInfectionState.DECEASED): additional_mortality
        for stage, additional_mortality
        in additional_mortalities_capped.items()
    }


def load_disease_durations(disease_durations_file: PathLike) -> dict[HPVGenotype, float]:
    """Loads HPV infection durations, in months, from a YAML file.
    
    Args:
        disease_durations_file (PathLike): The path to the YAML file containing the disease durations.
    
    Returns:
        dict[HPVGenotype, float]: A dictionary mapping HPV genotypes to their
        average durations.
    """
    with open(disease_durations_file, "r") as f:
        disease_durations = yaml.load(f, Loader=yaml.FullLoader)
    
    # convert genotype strings into HPVGenotype enums
    for genotype in HPVGenotype:
        if genotype.name not in disease_durations:
            raise ValueError(
                f"No duration found for {genotype.value}",
            )
        disease_durations[genotype] = disease_durations.pop(genotype.name)
    return disease_durations


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


def load_relative_incidences_by_age(
    incidences_by_age_file: PathLike,
    age_sexual_initiation: int = DEFAULT_AGE_FIRST_EXPOSURE,
) -> dict[float, float]:
    """Loads the relative incidence of HPV as a function of age.
    
    Args:
        incidences_by_age_file (PathLike):  The path to the CSV file
        containing the ages and their respective incidences.
    
    Returns:
        dict[int, float]: A dictionary mapping ages, in years, to their
        respective incidences, as a proportion of the initial incidence.
    """
    incidences_by_age = {}

    # read raw incidence rates by age from Muñoz et al. (2004)
    with open(incidences_by_age_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            # Skip the metadata comments and the header
            if row and not row[0].startswith("#") and not row[0] == "Age":  
                # age keys are approximated to the closest integers
                incidences_by_age[int(round(float(row[0])))] = float(row[1])
    
    # extrapolate the first age group to younger ages
    initial_incidence = incidences_by_age[min(incidences_by_age.keys())]
    incidences_by_age[0] = 0.0
    incidences_by_age[age_sexual_initiation] = initial_incidence

    # return all values as relative to the initial incidences
    return {k: (v / initial_incidence) for k, v in incidences_by_age.items()}


def estimate_age_adjusted_incidences(
    reference_incidence: float,
    reference_age: int | float,
    relative_incidences_by_age: dict[int, float],
) -> dict[int, float]:
    """Estimates the age-adjusted incidences of HPV.

    Args:
        reference_incidence (float): The incidence of HPV at the reference age.
        reference_age (int | float): The reference age for the given incidence
            estimate.
        relative_incidences_by_age (dict[int, float]): A dictionary mapping
            ages, in years, to their respective incidences, as a proportion
            of the initial incidence.

    Returns:
        dict[int, float]: A dictionary mapping ages, in years, to their
            respective incidences, adjusted for the expected changes in 
            incidence as age changes.
    """

    reference_age_key = max(
        k for k in relative_incidences_by_age.keys() if k <= reference_age
    )
    relative_incidence_reference_age = (
        relative_incidences_by_age[reference_age_key]
    )

    age_adjusted_incidences = {}
    for age, relative_incidence in relative_incidences_by_age.items():
        adjustment_factor = (
            relative_incidence / relative_incidence_reference_age
        )
        age_adjusted_incidences[age] = reference_incidence * adjustment_factor

    return age_adjusted_incidences


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
    disease_duration: float,
) -> float:  
    """
    Estimate the average incidence rate given prevalence and clearance rate.
    
    Args:
        prevalence (float): The disease prevalence in the population.
        disease_duration (float): The average disease duration, in months.

    Returns:
        float: Estimated incidence rate.
    """
    return prevalence / disease_duration

# Default yearly discount rate for the discounting benefits in the future
DEFAULT_DISCOUNT_RATE: float = 0.043


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

# Mode of the distribution of cancer prevalence in Brazil, 2021
# Source: https://vizhub.healthdata.org/gbd-results?
# params=gbd-api-2021-permalink/2845eaee2d516e046adbc01b8c9c8c91
MODE_CANCER_PREVALENCE: int = 37

# Transition probabilities for each HPV genotype
_epidemiological_params = load_epidemiological_parameters(
    natural_history_params_file=_NATURAL_HISTORY_PARAMS_FILE,
)

# Additional mortality for all 
_cancer_additional_mortality_rates = load_cancer_additional_mortalities(
    survival_curves_file=_SURVIVAL_CURVES_FILE,
    non_cancer_mortality_rates=NON_CERVICAL_CANCER_MORTALITY_RATES,
    mode_cancer_prevalence=MODE_CANCER_PREVALENCE,
    consider_detected_cancers_cured_at=CONSIDER_DETECTED_CANCERS_CURED_AT,
)
for genotype in _epidemiological_params:
    _epidemiological_params[genotype].update(
        _cancer_additional_mortality_rates
    )

TRANSITION_PROBABILITIES: dict[HPVGenotype, NDArray] = {
    genotype: _convert_param_dict_to_array(params)
    for genotype, params in _epidemiological_params.items()
}


# estimate age-adjusted incidences
_prevalences = load_prevalences(_PREVALENCES_FILE)
_disease_durations = load_disease_durations(_DISEASE_DURATION_FILE)
_reference_incidences = {
    genotype: estimate_incidence(
        prevalence=_prevalences[genotype],
        disease_duration=_disease_durations[genotype],
    )
    for genotype in HPVGenotype
}
_reference_incidences_age = 21  # approximate age of the POP-BRASIL study
_relative_incidences_by_age = load_relative_incidences_by_age(
    _INCIDENCES_BY_AGE_FILE,
)

INCIDENCES: dict[HPVGenotype, dict[int, float]] = {}
for genotype in HPVGenotype:
    INCIDENCES[genotype] = estimate_age_adjusted_incidences(
        reference_incidence=_reference_incidences[genotype],
        reference_age=_reference_incidences_age,
        relative_incidences_by_age=_relative_incidences_by_age,
    )

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
