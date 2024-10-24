# -*- coding: utf-8 -*-

"""Core model definitions for simulating HPV progression.

This module defines the core classes and methods used for simulating the
progression of HPV infections in individuals and cohorts over time. It 
provides mechanisms for tracking infection states, vaccination, screening, and
the effects of interventions.

The primary entities are:
    - `HPVInfection`: Represents an individual's HPV infection state over time.
    - `Individual`: Models a person in the simulation, tracking infections,
    screenings, 
      and transitions between health states.
    - `Cohort`: Represents a group of individuals, applying HPV exposure,
    screening, vaccination, and progression over time.
"""

from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import permutations

import numpy as np
from prettytable import PrettyTable
from scipy.stats import lognorm
from scipy.stats._continuous_distns import lognorm_gen
from scipy.optimize import minimize

from .common import (
    CANCER_STATES,
    HPVGenotype,
    HPVInfectionState,
    Longitudinal,
    ObservableOutcome,
    RNG,
    ScreeningMethod,
    ScreeningRegimen,
    Snapshot,
    UNDETECTED_CANCER_STATES
)
from .params import (
    NATURAL_IMMUNITY,
    NON_CERVICAL_CANCER_MORTALITY_RATES,
    QUADRIVALENT_EFFECTIVENESS,
    TRANSITION_PROBABILITIES,
    get_life_expectancy,
)


@lru_cache
def _fit_lognorm_to_compliance(
    compliance_rate: float,
    target_time: int,
) -> lognorm_gen:
    """Fits a log-normal curve to a compliance rate at the target_time.

    Args:
        compliance_rate (float): The target cumulative probability (e.g., 0.77
        for 77% compliance).
        target_time (int): The time (in months) at which the cumulative
        probability should equal the compliance rate.
    
    Returns:
        scipy.stats._continuous_distns.lognorm_gen: A log-normal distribution
        object with the parameters (mu, sigma) that fit the compliance rate at
        the target time.
    """
    # Define the loss function to minimize the difference between the CDF at target_time and compliance_rate
    def loss(params):
        mu, sigma = params
        cdf_value = lognorm.cdf(target_time, s=sigma, scale=np.exp(mu))
        return (cdf_value - compliance_rate) ** 2
    
    # Initial guesses for mu and sigma
    initial_guess = [np.log(target_time), 0.5]
    
    # Minimize the loss function to find the best-fit parameters
    result = minimize(loss, initial_guess, bounds=[(None, None), (1e-5, None)])
    
    mu, sigma = result.x
    return lognorm(s=sigma, scale=np.exp(mu))


class HPVInfection(Longitudinal):
    """Represents an HPV infection in an individual.

    This class models the infection dynamics of HPV genotypes, including transitions between different infection states over time based on
    transition probabilities.

    Args:
        genotype (HPVGenotype): The genotype of the HPV infection.

    Attributes:
        genotype (HPVGenotype): The genotype of the infection.
        current_state (HPVInfectionState): The current infection state.
        previous_state (HPVInfectionState): The infection state prior to the most recent transition.
        time_in_current_state (int): Number of months spent in the current infection state.
        transition_probabilities (np.ndarray): Transition probabilities between states based on time and genotype.
    """
    
    def __init__(
            self,
            genotype=HPVGenotype,
            initial_state=HPVInfectionState.INFECTED,
        ):
        self._t = 0
        self._genotype = genotype
        self._previous_state = None
        self._current_state = initial_state
        self._time_in_current_state = 0
        self._transition_probabilities = TRANSITION_PROBABILITIES[genotype]

    def next(self) -> None:
        """Advances the infection state based on transition probabilities."""
        self._previous_state = self.state
        transition_probabilities = self.transition_probabilities
        next_state = HPVInfectionState(RNG.choice(
            HPVInfectionState,
            size=1,
            p=transition_probabilities,
        ).item())
        if next_state != self.state:
            self._current_state = next_state
            self._time_in_current_state = 0
        else:
            self._time_in_current_state += 1
        self._t += 1

    @property
    def current_state(self) -> HPVInfectionState:
        """Returns the current infection state."""
        return self._current_state

    @property
    def genotype(self):
        """Returns the genotype of the infection."""
        return self._genotype

    @property
    def previous_state(self) -> HPVInfectionState:
        """Returns the previous infection state."""
        return self._previous_state

    @property
    def state(self) -> HPVInfectionState:
        """Alias for the `current_state` property."""
        return self.current_state

    @property
    def time_in_current_state(self) -> int:
        """Returns the number of months spent in the current infection state."""
        return self._time_in_current_state

    @property
    def transition_probabilities(self):
        """Returns the transition probabilities based on the current state and time spent in it."""
        return self._transition_probabilities[
            self.time_in_current_state,
            self.current_state,
            :,
        ]


class Individual(Longitudinal):
    """Represents an individual in the simulation.

    This class models an individual, including their HPV infections, screening history, 
    and vaccination status. It handles transitions between health states over time.

    Args:
        age (int | float): The age of the individual at the start of the simulation.

    Attributes:
        id_ (int): A unique identifier for the individual.
        age (float): The individual's current age in years.
        age_monhts (int): The individual's current age in months.
        infections (list[HPVInfection]): List of active HPV infections.
        immune_against (set[HPVGenotype]): Set of HPV genotypes the individual is immune to.
        time_since_last_screening (int): Number of months since the last screening.
        last_screening_age_months (int): Age of the last screening, in months.
        last_screening_result (bool): Result of the last screening.

    Example:

        ```py
        >>> from hpv_progression_model.model import Individual, HPVGenotype
        >>> individual = Individual(age=30)
        >>> individual.infections.add(HPVInfection(HPVGenotype.HPV_16))
        >>> individual.next()
        >>> individual.age
        30.083333333333332  # One month later
        >>> individual.current_state
        <HPVInfectionState.INFECTED: 0>
        ```
    """

    def __init__(
        self,
        age: int | float,
    ):
        self.id_: int = f"{abs(RNG.integers(10**12)):012d}"

        self._age_months: int = int(12 * age)
        self._t: int = 0
        self._non_cancer_death_probability = (
            NON_CERVICAL_CANCER_MORTALITY_RATES[0]
        )
        self._previous_state: HPVInfectionState | None = None
        self._state: HPVInfectionState = HPVInfectionState.HEALTHY

        self._is_vaccinated: bool = False
        self.infections: set[HPVInfection] = set()
        self.immune_against: set[HPVGenotype] = set()

        self._last_screening_age_months: int | None = None
        self._previous_screening_results: list[bool] = []
        self._screening_regimen: ScreeningRegimen | None = None
        self.screening_recommended_interval: int | None = None
        self.next_screening_method: ScreeningMethod | None = None

    def next(self):
        """Advances the individual's infection states and age by one month."""
        self._previous_state = self.state

        # Update the state of the individual
        if self.state != HPVInfectionState.DECEASED:
            infections_to_remove = set()
            for infection in self.infections:
                infection.next()
                # remove cleared infections and (possibly) add to immune list
                if infection.state == HPVInfectionState.HEALTHY:
                    if NATURAL_IMMUNITY >= RNG.random():
                        self.immune_against.add(infection.genotype)
                    infections_to_remove.add(infection)
            self.infections.difference_update(infections_to_remove)

            # Apply non-cancer related death probability
            if self.non_cancer_death_probability >= RNG.random():
                self.state = HPVInfectionState.DECEASED

            # Do not consider infections for deceased individuals
            if self.state == HPVInfectionState.DECEASED:
                self.infections.clear()

        self._age_months += 1
        self._t += 1

    def screen(self) -> bool:
        """Performs a screening for HPV-related lesions.

        The result depends on the individual's infection state, the screening
        method's sensitivity, and specificity. If the individual has
        undetected cancer, the cancer state may transition to detected after a
        positive screening result.

        Returns:
            bool: True if the screening detected HPV-related lesions, False otherwise.
        
        Raises:
            ValueError: If the screening method and age are not set.
        """
        if not self.next_screening_method:
            raise ValueError(
                "Screening method must be set before screening. "
                "Use set an appropriate value to the  `.screening_regimen` "
                "attribute to update the screening recommendation.",
            )

        if self.state == HPVInfectionState.DECEASED:
            raise ValueError(
                "Screening is not possible for deceased individuals.",
            )

        screening_result = None
        if self.current_state in (
            HPVInfectionState.HEALTHY,
            HPVInfectionState.INFECTED,
        ):
            # False-positives
            screening_result = (
                (1 - self.next_screening_method.specificity) >= RNG.random()
            )
        else:
            # True-positives
            screening_result = (
                self.next_screening_method.sensitivity.get(
                    self.current_state,
                    0,
                ) >= RNG.random()
            )
            if (
                screening_result
                and self.current_state in UNDETECTED_CANCER_STATES
            ):
                for infection in self.infections:
                    if infection.current_state in UNDETECTED_CANCER_STATES:
                        # transition to "detected" state
                        infection._current_state += 1
                        infection._time_in_current_state = 0

        self._last_screening_age_months = deepcopy(self.age_months)
        self._previous_screening_results.append(screening_result)
        self.update_screening_recommendation()
        return screening_result

    def see_and_treat_lesions(self):
        """Treats CIN 2 and CIN 3 lesions using the 'see and treat' approach.

        This method attempts to cure treatable lesions. If the treatment is successful, the HPV infection is cleared. Otherwise, the lesion is
        removed, but the infection remains active.
        """
        treatable_states = [HPVInfectionState.CIN2, HPVInfectionState.CIN3]
        for infection in self.infections:
            if infection.current_state in treatable_states:
                # remove lesion, but keep the infection active
                infection._current_state = HPVInfectionState.INFECTED
                infection._time_in_current_state = 0

                # NOTE: According to Arbyn et al., cited by INCA (2016),
                # section "Seguimento pós-tratamento de NIC II/III", p. 82
                # (available from: <https://www.inca.gov.br/publicacoes/livros
                # /diretrizes-brasileiras-para-o-rastreamento-do-cancer-do-
                # colo-do-utero>), 8% of the precancerous lesions reappear
                # in the following two years. This would be difficult to model
                # without an additional state, so what we do is to assume the
                # lesion is removed, but the infection remains active and can
                # progress again to CIN 2 or CIN 3 at the same rate as before.

    def update_screening_recommendation(self) -> None:
        """Updates the screening recommendation for the individual.

        This method uses the screening regimen followed by the individual and
        uses it to determine the age at which the screening will be performed
        next and the screening method to be used.
        """
        (
            self.next_screening_method,
            self.screening_recommended_interval,
        ) = self.screening_regimen.get_recommendation(
            age=self.age,
            last_screening_result=(
                self.previous_screening_results[-1]
                if self.previous_screening_results else None
            ),
        )

    def vaccinate_quadrivalent(self):
        """Administers the quadrivalent HPV vaccine to the individual.

        This method randomly determines if the individual becomes immune to
        specific HPV genotypes based on the vaccine's effectiveness.
        """
        for genotype, effectiveness in QUADRIVALENT_EFFECTIVENESS.items():
            if effectiveness >= RNG.random():
                self.immune_against.add(genotype)
        self._is_vaccinated = True

    @property
    def age(self) -> float:
        """Returns the individual's current age, in years."""
        return round(self._age_months / 12, 6)

    @property
    def age_months(self) -> int:
        """Returns the individual's current age, in months."""
        return self._age_months

    @property
    def current_state(self) -> HPVInfectionState:
        """Alias for the `Individual.state` property."""
        return self.state

    @property
    def is_vaccinated(self) -> bool:
        """Returns whether the individual is vaccinated."""
        return self._is_vaccinated

    @property
    def last_screening_age_months(self) -> int | None:
        """Returns the age at which the individual was last screened."""
        return self._last_screening_age_months
    
    @property
    def next_screening_age_months(self) -> int | None:
        """Returns the age at which the individual will next be screened."""
        if self.screening_recommended_interval is None:
            raise ValueError(
                "No screening recommendation has been set for the individual."
            )
        if self.last_screening_age_months is None:
            return int(12 * self.screening_regimen.start_age)
        return (
            self.last_screening_age_months
            + self.screening_recommended_interval
        )

    @property
    def non_cancer_death_probability(self) -> float:
        """Probability of death due to causes unrelated to cervical cancer."""
        if self.age in NON_CERVICAL_CANCER_MORTALITY_RATES:
            self._non_cancer_death_probability = (
                NON_CERVICAL_CANCER_MORTALITY_RATES[int(round(self.age))]
            )
        return self._non_cancer_death_probability

    @property 
    def previous_state(self) -> HPVInfectionState:
        """Returns the previous infection state."""
        return self._previous_state

    @property
    def previous_screening_results(self) -> list[bool]:
        """Returns the result of the last screening."""
        return self._previous_screening_results

    @property
    def screening_regimen(self) -> ScreeningRegimen:
        """Returns the screening regimen followed by the individual."""
        return self._screening_regimen

    @screening_regimen.setter
    def screening_regimen(self, screening_regimen: ScreeningRegimen) -> None:
        """Sets the screening regimen followed by the individual."""
        self._screening_regimen = screening_regimen
        self.update_screening_recommendation()

    @property
    def screening_is_due(self) -> bool:
        """Returns whether the screening is due for the individual.
        
        Raises:
            ValueError: If the screening regimen is not set.
        """
        if self.next_screening_age_months is None:
            raise ValueError(
                "No screening recommendation has been set for the individual."
            )
        return self.next_screening_age_months <= self.age_months

    @property
    def state(self) -> HPVInfectionState:
        """Returns the current infection state."""
        if self._state == HPVInfectionState.DECEASED:
            return self._state
        elif self.infections:
            self._state = max(
                infection.state
                for infection in self.infections
            )
        else:
            self._state = HPVInfectionState.HEALTHY
        return self._state

    @state.setter
    def state(self, state: HPVInfectionState) -> None:
        if state == HPVInfectionState.DECEASED:
            self._state = HPVInfectionState.DECEASED
        else:
            raise ValueError(
                "The only state that can be directly set is ",
                "`HPVInfectionState.DECEASED`; for all other states, "
                "use the `.next()` method to update the infection state.",
            )

    @property
    def time_since_last_screening(self) -> int:
        """Returns the number of months since the last screening."""
        if not self.last_screening_age_months:
            return np.inf
        return self.age_months - self.last_screening_age_months


class Cohort(Longitudinal):
    """Represents a group of individuals (a cohort) in the simulation.

    This class models a cohort of individuals, simulating exposure to HPV infections, 
    vaccinations, screenings, and transitions between health states over time.

    Args:
        age (int | float): Initial age of individuals in the cohort.
        num_individuals (int): Number of individuals in the cohort.
        incidences (dict[HPVGenotype, float]): Dictionary mapping HPV genotypes to incidence rates.
        quadrivalent_coverage (float): Proportion of the cohort vaccinated with the quadrivalent vaccine.
        screening_regimen (ScreeningRegimen): Screening regimen followed by individuals in the cohort.
        screening_compliance (float): Proportion of the cohort that is screened according to the regimen.
        screening_followup_loss (float): Proportion of individuals lost to follow-up after positive screening.

    Attributes:
        age (float): Current age of individuals in the cohort, in years.
        age_months (int): Current age of individuals in the cohort, in months.
        age_started (float): Age of the cohort at the start of the simulation.
        num_individuals (int): Number of individuals in the cohort.
        individuals (set[Individual]): Set of individuals in the cohort.
        incidences (dict[HPVGenotype, float]): Incidence rates of different HPV genotypes.
        quadrivalent_coverage (float): Vaccination coverage for the quadrivalent vaccine.
        screening_regimen (ScreeningRegimen): Screening regimen applied to the cohort.
        screening_compliance (float): Proportion of the cohort that receives screenings.
        screening_followup_loss (float): Proportion of individuals lost to follow-up.
        history (dict[int, dict[HPVInfectionState, int]]): Historical record of cohort states by time.
        outcomes (dict[int, dict[ObservableOutcome, int]]): Observable outcomes like screenings and detections by time.
        transitions (dict[int, dict[tuple[HPVInfectionState, HPVInfectionState], int]]): Transitions between infection states by time.

    Example:

        ```py
        >>> from hpv_progression_model.model import Cohort
        >>> cohort = Cohort(
                age=30, 
                num_individuals=10000, 
                incidences={HPVGenotype.HPV_16: 0.001}, 
                quadrivalent_coverage=0.8, 
                screening_regimen=PAP_SMEAR_3YRS_25_64, 
                screening_compliance=0.6, 
                screening_followup_loss=0.1
            )
        >>> cohort.next()
        >>> cohort.age
        30.083333333333332  # One month later
        >>> cohort.prevalences
        {<HPVInfectionState.INFECTED: 0>: 0.00098, <HPVInfectionState.HEALTHY: 1>: 0.99902}
        ```
    """

    def __init__(
        self,
        age: int | float,
        num_individuals: int,
        incidences_by_age: dict[HPVGenotype, dict[int, float]],
        quadrivalent_coverage: float,
        screening_regimen: ScreeningRegimen,
        screening_compliance: float,
        screening_followup_loss: float,
    ):

        # Initialize individuals in the cohort
        self.individuals = {
            Individual(age=age)
            for _ in range(num_individuals)
        }

        # Set attributes
        self._t: int = 0
        self._age_months: int = int(12 * age)
        self._age_started: float = float(age)
        self._incidences_by_age: dict[HPVGenotype, dict[int, float]] = (
            incidences_by_age
        )
        self.quadrivalent_coverage: float = quadrivalent_coverage
        self.screening_regimen: ScreeningRegimen = screening_regimen
        self.screening_compliance: float = screening_compliance
        self.screening_followup_loss: float = screening_followup_loss

        # Vaccinate individuals in the cohort based on coverage
        for individual in RNG.choice(
            list(self.individuals),
            size=int(num_individuals * self.quadrivalent_coverage),
            replace=False,
        ):
            individual.vaccinate_quadrivalent()

        # Initialize history, outcomes, and transitions
        self._history_states: dict[
            int,
            dict[HPVInfectionState, int],
        ] = {}
        self._update_history_states()
        self._history_transitions: dict[
            int,
            dict[tuple[HPVInfectionState, HPVInfectionState], int],
        ] = {}
        self._outcomes: dict[int, dict[ObservableOutcome, int]] = {}

    def __hash__(self) -> int:
        return hash(tuple(self.id_, self.t))

    def __repr__(self) -> str:
        return (
            f"Cohort(id={self.id_}, "
            f"num_individuals={self.num_individuals}, "
            f"age={self.age}, "
            f"t={self.t})"
        )

    def expose(self) -> None:
        """Exposes the cohort to new HPV infections.

        This method simulates HPV exposure in the cohort by randomly exposing individuals 
        to new infections based on the incidence rates of different HPV genotypes.
        """
        individuals_alive = [
            i for i in self.individuals
            if i.current_state != HPVInfectionState.DECEASED
        ]
        for genotype, incidence in self.incidences.items():
            exposed_individuals = RNG.choice(
                individuals_alive,
                size=int(self.num_individuals_alive * incidence),
                replace=False,
            )
            for individual in exposed_individuals:
                if (
                    genotype not in individual.immune_against
                    and all(
                        infection.genotype != genotype
                        for infection in individual.infections
                    )
                ):
                    individual.infections.add(HPVInfection(genotype))

    def next(self):
        """Advances the cohort simulation by one month.

        This method advances the simulation by:
        - Taking a snapshot of the cohort's current infection states.
        - Exposing the cohort to new HPV infections.
        - Screening eligible individuals.
        - Treating positive screenings (see-and-treat).
        - Updating outcomes and transitions.
        """

        # Advance the simulation for each individual in the cohort
        for individual in self.individuals:
            individual.next()

        # Expose the cohort to new infections
        self.expose()

        # Screen eligible individuals
        screening_positives = self.screen()

        # Treat positive screenings for precancer lesions
        self._outcomes[self.t] = defaultdict(int)
        for individual in screening_positives:
            if individual.state == HPVInfectionState.CIN2:
                self._outcomes[self.t][ObservableOutcome.CIN2_DETECTIONS] += 1
            if individual.state == HPVInfectionState.CIN3:
                self._outcomes[self.t][ObservableOutcome.CIN3_DETECTIONS] += 1
            if (1 - self.screening_followup_loss) >= RNG.random():
                self._outcomes[self.t][ObservableOutcome.COLPOSCOPIES] += 1
                if individual.state == HPVInfectionState.CIN2:
                    self._outcomes[self.t][
                        ObservableOutcome.EXCISIONS_TYPES_1_2
                    ] += 1
                    self._outcomes[self.t][ObservableOutcome.BIOPSIES] += 1
                if individual.state == HPVInfectionState.CIN3:
                    self._outcomes[self.t][
                        ObservableOutcome.EXCISIONS_TYPE_3
                    ] += 1
                    self._outcomes[self.t][ObservableOutcome.BIOPSIES] += 1
                if individual.state in CANCER_STATES:
                    self._outcomes[self.t][ObservableOutcome.BIOPSIES] += 1
                individual.see_and_treat_lesions()

        self._update_history_transitions()
        self._update_outcomes()

        # Advance the cohort's time and age
        self._age_months += 1
        self._t += 1
        self._update_history_states()

    def screen(self) -> set[Individual]:
        """Screens individuals in the cohort.

        This method determines which individuals perform the prescribed screening method, taking into account the expeted compliance rates, and returns a set of individuals who tested positive for HPV-related lesions. 

        Returns:
            set[Individual]: A set of individuals who tested positive in the screening.
        """
        individuals_positive_results = set()
        for individual in self.individuals:
            if individual.state == HPVInfectionState.DECEASED:
                continue
            recommended_interval = individual.screening_recommended_interval
            if individual.time_since_last_screening < np.inf:
                time_since_last_screening = (
                    individual.time_since_last_screening
                )
            else:
                time_since_last_screening = recommended_interval
            if (
                individual.age >= individual.screening_regimen.start_age and
                individual.age < individual.screening_regimen.start_age + 1/12
            ):
                # Assume individuals that complete the starting age for
                # screening comply in the first month, at the same rate as the
                # general compliance rate. This is a simplification.
                compliance_probability = self.screening_compliance
            else:
                compliance_probability = self._get_compliance_probability(
                    recommended_interval,
                    time_since_last_screening,
                )
            if compliance_probability >= RNG.random():
                screening_result = individual.screen()
                if screening_result:
                    individuals_positive_results.add(individual)
        return individuals_positive_results

    def take_snapshot(self) -> Snapshot["Cohort"]:
        """Takes a snapshot of the current state of the cohort.

        Returns:
            Snapshot: A snapshot of the current state of the cohort.
        """
        return Snapshot(self)

    def _get_compliance_probability(
        self,
        recommended_interval: int,
        time_since_last_screening: int,
    ) -> float:
        """Get the probability of being screened at a given month.

        Args:
            recommended_interval (int): The recommended interval for screening.
            time_since_last_screening (int): The number of months since the
            last screening.

        Returns:
            float: The probability of being screened at the given interval.
        """ 
        probability_distribution = _fit_lognorm_to_compliance(
            compliance_rate=self.screening_compliance,
            target_time=recommended_interval,
        )
        if time_since_last_screening < 1:
            # deal with the tail of the probability mass that comes from
            # negative or null times since last screening. In this case,
            # attribute all the probability mass to t=1
            return probability_distribution.cdf(time_since_last_screening)
        else:
            return probability_distribution.pdf(time_since_last_screening)

    def _update_history_states(self) -> None:
        """Adds the current infection states to the history.

        This method records the number of individuals in each infection state at the current time.

        Returns:
            dict[HPVInfectionState, int]: A dictionary mapping infection states to the number of 
            individuals in each state.
        """
        self._history_states[self.t] = {}
        for state in HPVInfectionState:
            num_individuals_in_state = len({
                individual.id_
                for individual in self.individuals
                if individual.current_state == state
            })
            self._history_states[self.t][state] = num_individuals_in_state

    def _update_history_transitions(self) -> None:
        """Adds the latest transitions between infection states to the history.

        This method updates the transition records whenever an individual's infection 
        transitions between states.
        """
        self._history_transitions[self.t] = {
            (state_source, state_destination): 0
            for state_source, state_destination in permutations(
                HPVInfectionState,
                r=2,
            )
        }
        for individual in self.individuals:
            if individual.previous_state != individual.current_state:
                self._history_transitions[self.t][(
                    individual.previous_state,
                    individual.current_state,
                )] += 1

    def _update_outcomes(self) -> None:
        """Updates the observable outcomes of the cohort.

        This method tracks observable outcomes such as the number of
        screenings, detections of CIN2 and CIN3, and detections of various
        stages of invasive cancer, and deaths.
        """
        outcomes = self._outcomes.pop(self.t, defaultdict(int))
        screened_individuals = {
            individual
            for individual in self.individuals
            if individual.time_since_last_screening == 0
        }
        local_detections = self.transitions[self.t][(
            HPVInfectionState.LOCAL_UNDETECTED,
            HPVInfectionState.LOCAL_DETECTED,
        )]
        regional_detections = self.transitions[self.t][(
            HPVInfectionState.REGIONAL_UNDETECTED,
            HPVInfectionState.REGIONAL_DETECTED,
        )]
        distant_detections = self.transitions[self.t][(
            HPVInfectionState.DISTANT_UNDETECTED,
            HPVInfectionState.DISTANT_DETECTED,
        )]
        deaths = sum(
            self.transitions[self.t][(state, HPVInfectionState.DECEASED)]
            for state in HPVInfectionState
            if state != HPVInfectionState.DECEASED
        )
        years_of_life_lost = deaths * get_life_expectancy(self.age)
        outcomes.update({
            ObservableOutcome.SCREENINGS: len(screened_individuals),
            ObservableOutcome.LOCAL_DETECTIONS: local_detections,
            ObservableOutcome.REGIONAL_DETECTIONS: regional_detections,
            ObservableOutcome.DISTANT_DETECTIONS: distant_detections,
            ObservableOutcome.DEATHS: deaths,
            ObservableOutcome.YLL: years_of_life_lost,
        })
        self._outcomes[self.t] = outcomes

    @property
    def age(self) -> float:
        return round(self._age_months / 12, 6)

    @property
    def age_months(self) -> int:
        return self._age_months

    @property
    def age_started(self) -> float:
        return self._age_started

    @property
    def history(self) -> dict[int, dict[HPVInfectionState, int]]:
        return self._history_states

    @property
    def id_(self) -> int:
        return hash(":".join(sorted([i.id_ for i in self.individuals])))
    
    @property
    def incidences(self) -> dict[HPVGenotype, float]:
        incidences: dict[HPVGenotype, float] = {}
        for genotype, incidences_by_age in self._incidences_by_age.items():
            last_age_with_incidence_data = max(
                k for k in incidences_by_age.keys() if k <= self.age
            )
            incidences[genotype] = (
                incidences_by_age[last_age_with_incidence_data]
            )
        return incidences

    @property
    def num_individuals(self) -> int:
        return len(self.individuals)

    @property
    def num_individuals_alive(self) -> int:
        return len({
            i for i in self.individuals
            if i.current_state != HPVInfectionState.DECEASED
        })

    @property
    def prevalences(self) -> dict[HPVInfectionState, float]:
        prevalences = {}
        for state in HPVInfectionState:
            if (
                state == HPVInfectionState.HEALTHY
                or state == HPVInfectionState.DECEASED
            ):
                continue
            if self.num_individuals_alive == 0:
                prevalences[state] = 0
            else:
                prevalences[state] = (
                    self.history[self.t][state] / self.num_individuals_alive
                )
        return prevalences

    @property
    def outcomes(self) -> dict[int, dict[ObservableOutcome, int]]:
        return self._outcomes

    @property
    def outcomes_accumulated(self) -> dict[ObservableOutcome, int]:
        return {
            outcome: sum(
                self.outcomes[t][outcome]
                for t in range(self.t)
            )
            for outcome in ObservableOutcome
        }

    @property
    def screening_regimen(self) -> ScreeningRegimen:
        """Returns the screening regimen applied to the cohort."""
        return self._screening_regimen
    
    @screening_regimen.setter
    def screening_regimen(self, screening_regimen: ScreeningRegimen) -> None:
        """Sets the screening regimen applied to the cohort."""
        self._screening_regimen = screening_regimen
        for individual in self.individuals:
            individual.screening_regimen = screening_regimen

    @property
    def summary(self) -> None:
        print(f"# Cohort ID #{self.id_}", end="\n\n")

        print("## Cohort information:", end="\n")
        cohort_info_table = PrettyTable(float_format="0.1")
        cohort_info_table.field_names = ["Variable", "Number"]
        cohort_info_table.add_rows([
            ["Number of individuals in the cohort", self.num_individuals],
            [
                "Initial age of the individuals in the cohort (years)",
                self.age_started,
            ],
            [
                "Current age of the individuals in the cohort (years)",
                self.age,
            ],
            ["Follow-up time (months)", self.t],
        ])
        print(cohort_info_table, end="\n\n")

        print("## Prevalence of infection states:", end="\n")
        prevalence_table = PrettyTable()
        prevalence_table.field_names = [
            "State",
            f"Prevalence at age {self.age:.0f} (%)",
        ]
        for state, prevalence in self.prevalences.items():
            prevalence_table.add_row([
                f"Prevalence: {state.name}",
                f"{prevalence:.2%}",
            ])
        print(prevalence_table, end="\n\n")

        print("## Accumulated outcomes since t=0:", end="\n")
        outcomes_table = PrettyTable()
        outcomes_table.field_names = ["Outcome", "# Events"]
        for outcome, total in self.outcomes_accumulated.items():
            if isinstance(total, float):
                total = f"{total:.2f}"
            outcomes_table.add_row([outcome, total])
        print(outcomes_table, end="\n\n")

    @property
    def transitions(self) -> dict[
        int,
        dict[tuple[HPVInfectionState, HPVInfectionState], int],
    ]:
        return self._history_transitions
