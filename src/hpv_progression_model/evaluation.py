# -*- coding: utf-8 -*-

"""Evaluation of simulation results for HPV progression.

This module provides classes and methods for evaluating the results of cohort simulations, including 
comparisons between interventions, calculating risk ratios, odds ratios, and other key metrics related 
to the effectiveness of screening, vaccination, and treatments. It also supports discounting outcomes 
to account for the time value of results.

Key classes include:
    - `DichotomousComparison`: Compares dichotomous outcomes (e.g., events vs. no events) between intervention and comparator groups.
    - `SimulationResults`: Encapsulates the baseline and endline cohort states after a simulation and allows for discounting of outcomes.
    - `Simulation`: Manages the execution of a cohort simulation over time and calculates resulting outcomes.

Attributes:
    DEFAULT_DISCOUNT_RATE (float): The default yearly discount rate applied to outcomes.
    MAX_FOLLOW_UP_DURATION (int): The maximum duration for following up a cohort simulation.

Example:
    Running a simulation and calculating results:

    ```py
    >>> from hpv_progression_model.evaluation import Simulation
    >>> from hpv_progression_model.model import Cohort
    >>> cohort = Cohort(age=30, num_individuals=1000, incidences={HPVGenotype.HPV_16: 0.1}, ...)
    >>> simulation = Simulation(cohort, interval=60)
    >>> simulation.run()
    >>> results = simulation.results
    ```

Todo:
    * Implement `DichotomousComparison` metrics such as odds ratio and risk difference.
    * Implement `summary` methods for concise result reporting.
"""

import sys
from collections.abc import Callable, defaultdict
from copy import deepcopy
from functools import contextmanager
from typing import Any, Generator

from prettytable import PrettyTable

from .common import (
    HPVGenotype,
    Immutable,
    ObservableOutcome,
    ScreeningRegimen,
    Snapshot,
)
from .model import Cohort
from .params import (
    DEFAULT_AGE_FIRST_EXPOSURE,
    DEFAULT_DISCOUNT_RATE,
    MAX_FOLLOW_UP_DURATION,
)


class DichotomousComparison(Immutable):
    """Compares dichotomous outcomes of an intervention and a comparator group.

    This class calculates key metrics such as risk ratio, odds ratio, risk
    difference, and number needed to treat based on the number of events and
    non-events in both the intervention and comparator groups.

    Args:
        intervention_events (int): Number of events in the intervention group.
        intervention_no_events (int): Number of non-events in the intervention group.
        comparator_events (int): Number of events in the comparator group.
        comparator_no_events (int): Number of non-events in the comparator group.

    Attributes:
        odds_ratio (float): The odds ratio between the two groups.
        risk_ratio (float): The risk ratio between the intervention and
        comparator groups.
        risk_difference (float): The absolute difference in risk between the
        two groups.
        relative_risk_reduction (float): The relative reduction in risk due to
        the intervention.
        number_needed_to_treat (int): The number needed to treat to prevent
        one additional event.
    """
    def __init__(
        self,
        intervention_events: int,
        intervention_no_events: int,
        comparator_events: int,
        comparator_no_events: int,
    ):

        intervention_sample_size = intervention_events + intervention_no_events
        comparator_sample_size = comparator_events + comparator_no_events
        
        # Calculate risk for intervention and comparator groups
        intervention_risk = intervention_events / intervention_sample_size
        comparator_risk = comparator_events / comparator_sample_size
        
        # Risk Ratio
        self.risk_ratio: float = (
            intervention_risk / comparator_risk
            if comparator_risk > 0 else float("inf")
        )
        
        # Odds Ratio
        intervention_odds = (
            intervention_events / intervention_no_events
            if intervention_no_events > 0 else float("inf")
        )
        comparator_odds = comparator_events / comparator_no_events if comparator_no_events > 0 else float("inf")
        self.odds_ratio: float = (
            intervention_odds / comparator_odds
            if comparator_odds > 0 else float("inf")
        )
        
        # Risk Difference
        self.risk_difference: float = intervention_risk - comparator_risk
        
        # Relative Risk Reduction
        self.relative_risk_reduction: float = (
            (1 - self.risk_ratio)
            if self.risk_ratio < 1 else 0
        )

        # Number Needed to Treat (NNT) (inverse of absolute risk reduction)
        self.number_needed_to_treat: int = (
            int(1 / self.absolute_risk_reduction)
            if self.absolute_risk_reduction > 0 else float("inf")
        )

    @property
    def summary(self) -> None:
        """Print summary tables with the comparison results."""

        print("## Comparison information:", end="\n")
        comparison_info_table = PrettyTable(float_format="0.1")
        comparison_info_table.field_names = ["Effect measure", "Value"]
        comparison_info_table.add_rows([
            ["Odds Ratio (OR)", self.odds_ratio],
            ["Risk Ratio (RR)", self.risk_ratio],
            ["Risk Difference (RD)", self.risk_difference],
            ["Relative Risk Reduction (RRR)", self.relative_risk_reduction],
            ["Number Needed to Treat (NNT)", self.number_needed_to_treat],
        ])


class SimulationResults(Immutable):
    """Encapsulates the results of a cohort simulation.

    This class stores the baseline and endline cohort states and provides methods 
    to discount outcomes over time to account for the time value of events. It also 
    aggregates events and non-events over the simulation period.

    Args:
        baseline (Snapshot[Cohort]): The cohort at the start of the simulation.
        endline (Snapshot[Cohort]): The cohort at the end of the simulation.

    Attributes:
        baseline (Snapshot[Cohort]): The cohort at baseline.
        endline (Snapshot[Cohort]): The cohort at the endline.
        is_discounted (bool): Whether the results have been discounted.
        discount_rate (float): The yearly discount rate applied to outcomes.

    """

    def __init__(
        self,
        baseline: Snapshot[Cohort],
        endline: Snapshot[Cohort],
    ):
        assert baseline.id_ == endline.id_, (
            "Baseline and endline must be the same cohort in different points "
            "in time, but the cohorts provided have different IDs."
        )
        assert endline.t > baseline.t, (
            "The endline must be in a time point after the baseline, but the "
            f"endline is on t={endline.t} and the baseline is on "
            f"t={baseline.t}."
        )
        self.baseline: Snapshot[Cohort] = baseline
        self.endline: Snapshot[Cohort] = endline
        self.is_discounted: bool = False
        self.discount_rate: float = 0.0

    def discount_outcomes(
        self,
        discount_rate_yearly: float = DEFAULT_DISCOUNT_RATE,
    ) -> None:
        """Applies a discount rate to the observable outcomes over time.

        This method discounts the outcomes of the simulation to account for the time 
        value of events. Discounting is done monthly based on the yearly discount rate.

        Args:
            discount_rate_yearly (float): The yearly discount rate (default is set by DEFAULT_DISCOUNT_RATE).

        """
        discount_rate_monthly = (1 + discount_rate_yearly) ** (1 / 12)
        for outcome in ObservableOutcome:
            for t in range(self.baseline.t, self.endline.t + 1):
                self.endline.outcomes[t][outcome] /= (
                    (1 + discount_rate_monthly) ** (t - self.baseline.t)
                )
        self.is_discounted = True
        self.discount_rate = discount_rate_yearly

    @property
    def events(self) -> dict[ObservableOutcome, int]:
        """Aggregates the number of events for each observable outcome.

        Returns:
            dict[ObservableOutcome, int]: A dictionary of cumulative event counts for each outcome.
        """
        events = defaultdict(int)
        for outcome in ObservableOutcome:
            for t in range(self.baseline.t, self.endline.t + 1):
                events[outcome] += self.endline.outcomes[t][outcome]
        return events

    @property
    def no_events(self) -> dict[ObservableOutcome, int]:
        """Calculates the number of non-events (complement of events) for each outcome.

        Returns:
            dict[ObservableOutcome, int]: A dictionary of the number of non-events for each outcome.
        """
        no_events = {}
        for outcome, events in self.events.items():
            no_events[outcome] = self.endline.num_individuals - events
        return no_events

    @property
    def summary(self) -> None:
        """Print summary tables with the simulation results."""

        print(
            f"# Simulation results for Cohort #{self.baseline.id_}, "
            f"ages {self.baseline.age} to {self.endline.age}: ",
            end="\n\n",
        )

        print("## Simulation information:", end="\n")
        simulation_info_table = PrettyTable(float_format="0.1")
        simulation_info_table.field_names = ["Attribute", "Value"]
        simulation_info_table.add_rows([
            ["Number of individuals", self.baseline.num_individuals],
            ["Age at baseline", self.baseline.age],
            ["Age at endline", self.endline.age],
            ["Time at baseline", self.baseline.t],
            ["Time at endline", self.endline.t],
            ["Discount rate", self.discount_rate],
        ])

        print("## Outcomes:", end="\n")
        outcomes_table = PrettyTable()
        outcomes_table.field_names = [
            "Outcome",
            "Baseline",
            "Endline",
            "Difference",
        ]
        for outcome in ObservableOutcome:
            outcomes_table.add_row([
                outcome.value,
                self.baseline.outcomes[self.baseline.t][outcome],
                self.endline.outcomes[self.endline.t][outcome],
                self.events[outcome],
            ])
        print(outcomes_table, end="\n\n")



class Simulation(object):
    """Runs a cohort simulation over a specified time interval.

    This class manages the execution of a cohort simulation, simulating the progression of 
    HPV infections, screenings, and treatments over time, starting at a specified age.

    Args:
        cohort (Cohort): The cohort of individuals to simulate.
        interval (int): The number of months over which to run the simulation.
        age_start (int | float | None): The starting age of the cohort. If not provided, 
            the simulation will begin at the cohort's current age.

    Attributes:
        cohort_original (Cohort): A deep copy of the original cohort at the start of the simulation.
        age_start (float): The starting age of the cohort.
        interval (int): The number of months over which the simulation is run.
        cohort_baseline (Cohort): A deep copy of the cohort after the warm-up period.
        cohort_endline (Cohort): The cohort after the simulation has completed.
    """

    def __init__(
        self,
        cohort: Cohort,
        interval: int,
        age_start: int | float | None = None,
    ):
        self.interval: int = interval
        self.cohort_original: Snapshot[Cohort] = cohort.take_snapshot()
        self.age_start: float = float(age_start or self.cohort_original.age)

        self._cohort_working_copy: Cohort = deepcopy(self.cohort_original)
        if self._cohort_working_copy.age < self.age_start:
            self.warm_up()
        self.cohort_baseline: Snapshot[Cohort] = (
            self._cohort_working_copy.take_snapshot()
        )

    def run(self) -> None:
        """Runs the cohort simulation for the specified interval.

        This method progresses the cohort simulation by advancing the cohort through time 
        for the defined number of months (`interval`).
        """
        t_start = self.cohort_baseline.t
        self.t = t_start
        while self.t < t_start + self.interval:
            self._cohort_working_copy.next()
        self.endline: Snapshot[Cohort] = (
            self._cohort_working_copy.take_snapshot()
        )

    def warm_up(self) -> None:
        """Advances the cohort to the starting age of the simulation.

        This method "warms up" the cohort by simulating its progression month by month 
        until the cohort reaches the specified starting age (`age_start`). It ensures 
        the cohort is at the appropriate age before the main simulation begins.
        """
        while self._cohort_working_copy.age < self.age_start:
            self._cohort_working_copy.next()

    @property
    def results(self) -> SimulationResults:
        """Returns the results of the simulation.

        This property generates and returns a `SimulationResults` object, encapsulating 
        the cohort's baseline (pre-simulation) and endline (post-simulation) states.

        Returns:
            SimulationResults: The results of the cohort simulation.
        """
        return SimulationResults(self.cohort_baseline, self.cohort_endline)

    @property
    def summary(self) -> None:
        """Generates a summary of the simulation."""
        raise NotImplementedError

@contextmanager
def apply_treatment(
    cohort: Cohort,
    t_start: int,
    duration: int = sys.maxsize,
    **modified_variables: dict[str, Any],
) -> Generator[Cohort, None, None]:
    """Context manager for applying a treatment to a cohort.

    Args:
        cohort (Cohort): The cohort to be treated.
        t_start (int): The start time of the treatment.
        lenght (int, optional): The length of the treatment. Defaults to `sys.maxsize`, which means the treatment will continue until the end of the cohort.
        **modified_variables: Additional keyword arguments with the attributes to be changed during the treatment in the cohort. See the documentation for [`hpv_progression_model.model.Cohort`](#hpv_progression_model.model.Cohort) all the modifiable attributes available.

    Yields:
        Cohort: The treated cohort.
    """
    t_end: int = t_start + duration
    original_attributes: dict[str, Any] = {}
    original_next_method: Callable[[Cohort], None] = cohort.next
    for attr_name in modified_variables.items():
        original_attributes[attr_name] = getattr(cohort, attr_name)
    
    def modified_next_method(self):
        if self.t >= t_start and self.t < t_end:
            for attr_name, attr_value in modified_variables.items():
                setattr(self, attr_name, attr_value)
        original_next_method(self)

    cohort.next = modified_next_method
    try:
        yield cohort
    finally:
        cohort.next = original_next_method
        for attr_name, attr_value in original_attributes.items():
            setattr(cohort, attr_name, attr_value)


def compare_results(
    intervention: SimulationResults,
    comparator: SimulationResults,
) -> dict[ObservableOutcome, DichotomousComparison]:
    """Compares the simulation results between an intervention and a comparator.

    This function calculates dichotomous comparisons (e.g., risk ratios, odds ratios) for each observable outcome 
    between two simulation results: one representing the intervention group and the other the comparator group.

    Args:
        intervention (SimulationResults): The simulation results for the intervention group.
        comparator (SimulationResults): The simulation results for the comparator group.

    Returns:
        dict[ObservableOutcome, DichotomousComparison]: A dictionary mapping each observable outcome to a 
        `DichotomousComparison` object that compares the intervention and comparator results.
    
    Example:
        ```py
        >>> from hpv_progression_model.evaluation import compare_results
        >>> comparison = compare_results(intervention_results, comparator_results)
        >>> comparison[ObservableOutcome.CIN2_DETECTIONS].risk_ratio
        0.75
        ```
    """
    comparisons = {}
    for outcome in ObservableOutcome:
        comparisons[outcome] = DichotomousComparison(
            intervention.events,
            intervention.no_events,
            comparator.events,
            comparator.no_events,
        )
    return comparisons


def evaluate_intervention(
    sample_size: int,
    incidences: dict[HPVGenotype, float],
    base_quadrivalent_coverage: float,
    base_screening_regimen: ScreeningRegimen,
    base_screening_compliance: float,
    base_screening_followup_loss: float,
    treatment_quadrivalent_coverage: float | None = None,
    treatment_screening_regimen: ScreeningRegimen | None = None,
    treatment_screening_compliance: float | None = None,
    treatment_screening_followup_loss: float | None = None,
    treatment_target_age: int | float | None = 25,
    treatment_duration: int = sys.maxsize,
    intervention_followup_duration: int = MAX_FOLLOW_UP_DURATION,
    age_first_exposure: int | float = DEFAULT_AGE_FIRST_EXPOSURE,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> SimulationResults | tuple[SimulationResults, SimulationResults]:
    """Evaluates the impact of an intervention compared to a base scenario.

    This function simulates the impact of an intervention compared to a base scenario (comparator). 
    The intervention can differ from the base scenario in terms of vaccination coverage, screening 
    regimen, screening coverage, and follow-up loss. It can also apply a discount rate to outcomes.

    Args:
        sample_size (int): The number of individuals in the cohort.
        incidences (dict[HPVGenotype, float]): The incidence rates of HPV genotypes in the cohort.
        base_quadrivalent_coverage (float): The quadrivalent vaccine coverage in the base scenario.
        base_screening_regimen (ScreeningRegimen): The screening regimen in the base scenario.
        base_screening_compliance (float): The screening coverage in the base scenario.
        base_screening_followup_loss (float): The proportion of individuals lost to follow-up in the base scenario.
        treatment_quadrivalent_coverage (float | None): The quadrivalent vaccine coverage in the treatment group (optional).
        treatment_screening_regimen (ScreeningRegimen | None): The screening regimen in the treatment group (optional).
        treatment_screening_compliance (float | None): The screening coverage in the treatment group (optional).
        treatment_screening_followup_loss (float | None): The proportion of individuals lost to follow-up in the treatment group (optional).
        treatment_target_age (int | float | None): The age at which the intervention starts. Defaults to 25.
        treatment_duration (int): The duration of the treatment intervention in months. Defaults to unlimited duration.
        intervention_followup_duration (int): The duration of the follow-up after the intervention in months.
        age_first_exposure (int | float): The age at which the cohort is first exposed to HPV. Defaults to
        `hpv_progression_model.params.DEFAULT_AGE_FIRST_EXPOSURE`.
        discount_rate (float): The yearly discount rate applied to the outcomes. Defaults to
        `hpv_progression_model.params.DEFAULT_DISCOUNT_RATE`.

    Returns:
        SimulationResults | tuple[SimulationResults, SimulationResults]: The results of the intervention and comparator simulations.
        If a discount rate is applied, returns both undiscounted and discounted comparisons.

    Example:
        ```py
        >>> from hpv_progression_model.evaluation import evaluate_intervention
        >>> results = evaluate_intervention(
                sample_size=1000,
                age_first_exposure=30,
                incidences={HPVGenotype.HPV_16: 0.1},
                base_quadrivalent_coverage=0.6,
                base_screening_regimen=PAP_SMEAR_3YRS_25_64,
                base_screening_compliance=0.8,
                base_screening_followup_loss=0.05
            )
        >>> undiscounted, discounted = results
        ```
    """
    
    modified_variables = {}
    for arg_name, arg_value in locals().items():
        if arg_name.startswith("treatment_"):
            if arg_value is None:
                locals()[arg_name] = locals()[
                    arg_name.replace("treatment_", "base_")
                ]
            else:
                modified_variables[arg_name.replace("treatment_", "")] = arg_value
    
    base_cohort = Cohort(
        age=age_first_exposure,
        num_individuals=sample_size,
        incidences=incidences,
        quadrivalent_coverage=base_quadrivalent_coverage,
        screening_regimen=base_screening_regimen,
        screening_compliance=base_screening_compliance,
        screening_followup_loss=base_screening_followup_loss,
    )
    treatment_start = (
        base_cohort.t + (treatment_target_age - base_cohort.age) * 12
    )
    
    # Applying treatment variables to the cohort
    with apply_treatment(
        base_cohort,
        t_start=treatment_start,
        duration=treatment_duration,
        **modified_variables,
    ) as treatment_cohort:
        treatment_simulation = Simulation(
            cohort=treatment_cohort,
            interval=intervention_followup_duration,
            age_start=treatment_target_age,
        )
        treatment_simulation.run()
        treatment_results = treatment_simulation.results
    
    comparator_simulation = Simulation(
        cohort=base_cohort,
        interval=intervention_followup_duration,
        age_start=treatment_target_age,
    )
    comparator_simulation.run()
    comparator_results = comparator_simulation.results

    # Compare results between intervention and comparator
    undiscounted_comparison = compare_results(
        intervention=treatment_results,
        comparator=comparator_results,
    )

    # If discounting is applied, discount outcomes and compare again
    if discount_rate and discount_rate > 0:
        treatment_results.discount_outcomes(discount_rate)
        comparator_results.discount_outcomes(discount_rate)
        discounted_comparison = compare_results(
            intervention=treatment_results,
            comparator=comparator_results,
        )
        return undiscounted_comparison, discounted_comparison

    return undiscounted_comparison
