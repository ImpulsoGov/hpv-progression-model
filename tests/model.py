from copy import deepcopy
from sys import float_info

import numpy as np
import pytest

from hpv_progression_model.common import (
    HPVGenotype,
    HPVInfectionState,
    ObservableOutcome,
    UNDETECTED_CANCER_STATES,
)
from hpv_progression_model.model import Cohort, HPVInfection, Individual
from hpv_progression_model.params import (
    MAX_FOLLOW_UP_DURATION,
    NATURAL_HISTORY,
    PAP_SMEAR,
    PAP_SMEAR_3YRS_25_64,
    TRANSITION_PROBABILITIES,
)


@pytest.fixture
def mock_rng_always_zero(monkeypatch):
    class MockRNG(np.random.Generator):
        def __init__(self, *args, **kwargs):
            super().__init__(np.random.PCG64(), *args, **kwargs)

        def random(self):
            return 0.0

    monkeypatch.setattr("hpv_progression_model.model.RNG", MockRNG())


@pytest.fixture
def mock_rng_always_one(monkeypatch):
    class MockRNG(np.random.Generator):
        def __init__(self, *args, **kwargs):
            super().__init__(np.random.PCG64(), *args, **kwargs)

        def random(self):
            return 1 + float_info.min

    monkeypatch.setattr("hpv_progression_model.model.RNG", MockRNG())


@pytest.fixture
def mock_transition_probabilities_no_change(monkeypatch):
    mock_transition_probabilities = np.zeros((
        MAX_FOLLOW_UP_DURATION,
        len(HPVInfectionState),
        len(HPVInfectionState),
    ))
    for state in HPVInfectionState: 
        t = 0
        while t < MAX_FOLLOW_UP_DURATION:
            mock_transition_probabilities[t, state, state] = 1.0
            t += 1
    monkeypatch.setattr(
        "hpv_progression_model.model.TRANSITION_PROBABILITIES",
        {genotype: mock_transition_probabilities for genotype in HPVGenotype},
    )


@pytest.fixture
def infection():
    return HPVInfection(genotype=HPVGenotype.HPV_16)


@pytest.fixture
def individual():
    # Initialize Individual with an example age
    return Individual(age=30)


@pytest.fixture
def individual_with_screening_regimen():
    individual = Individual(age=30)
    individual.screening_regimen = PAP_SMEAR_3YRS_25_64
    return individual


@pytest.fixture
def changeless_infection(mock_transition_probabilities_no_change):
    return HPVInfection(genotype=HPVGenotype.HPV_16)

@pytest.fixture
def cohort():
    return Cohort(
        age=15,
        num_individuals=3000,
        incidences_by_age={HPVGenotype.HPV_16: {0: 0.0, 15: 0.01}},
        quadrivalent_coverage=0.75,
        screening_regimen=PAP_SMEAR_3YRS_25_64,
        screening_compliance=0.8,
        screening_followup_loss=0.05,
    )

@pytest.fixture
def cohort_without_screening():
    return Cohort(
        age=15,
        num_individuals=3000,
        incidences_by_age={HPVGenotype.HPV_16: {0: 0.0, 15: 0.01}},
        quadrivalent_coverage=0.75,
        screening_regimen=NATURAL_HISTORY,
        screening_compliance=0.0,
        screening_followup_loss=1.0,
    )

@pytest.fixture
def cohort_low_vaccination():
    return Cohort(
        age=15,
        num_individuals=3000,
        incidences_by_age={HPVGenotype.HPV_16: {0: 0.0, 15: 0.01}},
        quadrivalent_coverage=0.05,
        screening_regimen=PAP_SMEAR_3YRS_25_64,
        screening_compliance=0.8,
        screening_followup_loss=0.05,
    )


class TestHPVInfection:
    @pytest.mark.parametrize("genotype", HPVGenotype)
    def test_initialization(self, genotype):
        infection = HPVInfection(genotype=genotype)
        assert infection.genotype == genotype
        assert infection.current_state == HPVInfectionState.INFECTED
        assert infection.previous_state is None
        assert infection.time_in_current_state == 0
        assert np.array_equal(
            infection.transition_probabilities,
            TRANSITION_PROBABILITIES[genotype][0][HPVInfectionState.INFECTED]
        )
        np.testing.assert_approx_equal(
            np.sum(infection.transition_probabilities),
            1.0,
        )

    def test_next_state_change(self, infection):
        initial_state = infection.current_state
        initial_t = infection.t
        while infection.state == initial_state:
            infection.next()
        assert infection.t >= initial_t
        assert infection.previous_state == initial_state
        assert infection.current_state != initial_state
        assert infection.time_in_current_state == 0

    def test_next_state_no_change(self, changeless_infection):
        infection = changeless_infection
        initial_state = infection.current_state
        assert infection.transition_probabilities[
            HPVInfectionState.INFECTED,
        ] == 1.0
        np.testing.assert_approx_equal(
            np.sum(infection.transition_probabilities),
            1.0,
        )
        for _ in range(200):
            infection.next()
        assert infection.t == 200
        assert infection.current_state == initial_state
        assert infection.previous_state == infection.current_state
        assert infection.time_in_current_state == 200

    def test_properties(self, infection):
        assert isinstance(infection.genotype, HPVGenotype)
        assert isinstance(infection.current_state, HPVInfectionState)
        assert isinstance(infection.previous_state, (HPVInfectionState, type(None)))
        assert isinstance(infection.state, HPVInfectionState)
        assert infection.current_state == infection.state
        assert isinstance(infection.time_in_current_state, int)
        assert isinstance(infection.transition_probabilities, np.ndarray)


class TestIndividual:
    def test_initialization(self, individual):
        assert individual.age == 30.0
        assert individual.current_state == HPVInfectionState.HEALTHY
        assert individual.infections == set()
        assert individual.immune_against == set()
        assert not individual.is_vaccinated

    def test_next_month(self, individual):
        initial_age = individual.age
        individual.next()
        assert individual.age == pytest.approx(initial_age + 1/12)
        assert individual.t == 1
    
    @pytest.mark.parametrize(
        "initial_state",
        [
            HPVInfectionState.INFECTED,
            HPVInfectionState.CIN2,
            HPVInfectionState.CIN3,
            HPVInfectionState.LOCAL_UNDETECTED,
            HPVInfectionState.REGIONAL_UNDETECTED,
            HPVInfectionState.DISTANT_UNDETECTED,
        ],
    )
    def test_changes_state(
        self,
        individual,
        infection,
        initial_state,
    ):
        infection._current_state = initial_state
        individual.infections.add(infection)
        initial_age = individual.age
        assert individual.state == initial_state
        assert individual.current_state == initial_state
        while individual.state == initial_state:
            individual.next()
        assert individual.previous_state == initial_state
        assert individual.current_state != initial_state
        assert individual.t > 0
        assert individual.age > initial_age

    @pytest.mark.usefixtures("mock_rng_always_zero")
    def test_vaccination_effective(self, individual):
        assert not individual.is_vaccinated
        individual.vaccinate_quadrivalent()
        assert individual.is_vaccinated
        assert individual.immune_against == {
            HPVGenotype.HPV_16,
            HPVGenotype.HPV_18,
        }

    @pytest.mark.usefixtures("mock_rng_always_one")
    def test_vaccination_ineffective(self, individual):
        assert not individual.is_vaccinated
        individual.vaccinate_quadrivalent()
        assert individual.is_vaccinated
        assert individual.immune_against == set()

    def test_screen(
        self,
        individual_with_screening_regimen,
    ):
        individual = individual_with_screening_regimen
        assert individual.screening_is_due
        assert individual.next_screening_age_months == int(25.0 * 12)
        assert individual.next_screening_method == PAP_SMEAR
        individual.screen()
        assert individual.next_screening_age_months == int(33.0 * 12)
        assert individual.next_screening_method == PAP_SMEAR

    @pytest.mark.usefixtures("mock_rng_always_zero")
    @pytest.mark.parametrize(
        "initial_state",
        [
            HPVInfectionState.CIN2,
            HPVInfectionState.CIN3,
            HPVInfectionState.LOCAL_UNDETECTED,
            HPVInfectionState.REGIONAL_UNDETECTED,
            HPVInfectionState.DISTANT_UNDETECTED,
        ],
    )
    def test_screening_effective(
        self,
        individual_with_screening_regimen,
        infection,
        initial_state,
    ):
        individual = individual_with_screening_regimen
        infection._current_state = initial_state
        individual.infections.add(infection)
        screening_was_positive = individual.screen()
        assert screening_was_positive
        if initial_state in UNDETECTED_CANCER_STATES:
            assert individual.state == initial_state + 1

    @pytest.mark.usefixtures("mock_rng_always_one")
    @pytest.mark.parametrize(
        "initial_state",
        [
            HPVInfectionState.CIN2,
            HPVInfectionState.CIN3,
            HPVInfectionState.LOCAL_UNDETECTED,
            HPVInfectionState.REGIONAL_UNDETECTED,
            HPVInfectionState.DISTANT_UNDETECTED,
        ],
    )
    def test_screening_ineffective(
        self,
        individual_with_screening_regimen,
        infection,
        initial_state,
    ):
        individual = individual_with_screening_regimen
        infection._current_state = initial_state
        individual.infections.add(infection)
        screening_was_positive = individual.screen()
        assert not screening_was_positive
        assert individual.state == initial_state

    def test_screening_without_method_raises_value_error(self, individual):
        with pytest.raises(ValueError):
            individual.screen()

    @pytest.mark.usefixtures("mock_rng_always_zero")
    @pytest.mark.parametrize(
        "state",
        [HPVInfectionState.CIN2, HPVInfectionState.CIN3],
    )
    def test_see_and_treat_effective(self, individual, infection, state):
        infection._current_state = state
        individual.infections.add(infection)
        assert individual.state == state
        individual.see_and_treat_lesions()
        assert individual.state == HPVInfectionState.INFECTED


    @pytest.mark.usefixtures("mock_rng_always_one")
    @pytest.mark.parametrize(
        "state",
        [HPVInfectionState.CIN2, HPVInfectionState.CIN3],
    )
    def test_see_and_treat_ineffective(self, individual, infection, state):
        infection._current_state = state
        individual.infections.add(infection)
        assert individual.state == state
        individual.see_and_treat_lesions()
        assert individual.state == HPVInfectionState.INFECTED

    @pytest.mark.usefixtures("mock_rng_always_zero")
    @pytest.mark.parametrize(
        "state",
        [
            HPVInfectionState.LOCAL_UNDETECTED,
            HPVInfectionState.REGIONAL_UNDETECTED,
            HPVInfectionState.DISTANT_UNDETECTED,
        ],
    )
    def test_see_and_treat_not_immediately_treatable(
        self,
        individual,
        infection,
        state,
    ):
        infection._current_state = state
        individual.infections.add(infection)
        assert individual.state == state
        individual.see_and_treat_lesions()
        assert individual.state == state  # treatment is not applicable


class TestCohort:
    def test_initialization(self, cohort):
        assert cohort.age == 15
        assert cohort.num_individuals == 3000
        assert len(cohort.individuals) == 3000
        assert cohort.history[0][HPVInfectionState.HEALTHY] == 3000

    def test_exposure(self, cohort):
        num_initially_infected = sum(
            1 for i in cohort.individuals if i.infections
        )
        for _ in range(100):  # Simulate multiple exposures
            cohort.expose()
        num_after_exposure = sum(
            1 for i in cohort.individuals if i.infections
        )
        assert num_after_exposure > num_initially_infected

    def test_vaccination_coverage(self, cohort):
        num_vaccinated = sum(
            1 for i in cohort.individuals if i.is_vaccinated
        )
        expected_vaccinated = int(0.75 * 3000)
        assert num_vaccinated == expected_vaccinated

    def test_screening(self, cohort_without_screening):
        cohort = cohort_without_screening
        for _ in range(120):  # Ensure some infections and minimal age
            cohort.next()
        cohort.screening_regimen = PAP_SMEAR_3YRS_25_64
        cohort.screening_compliance = 1.0 + float_info.min
        assert any(i.screening_is_due for i in cohort.individuals)
        screening_positives = cohort.screen()
        assert all(
            i.time_since_last_screening == 0
            or i.state == HPVInfectionState.DECEASED 
            for i in cohort.individuals
        )
        assert isinstance(screening_positives, set)
        assert len(screening_positives) > 0
        # assert there are some true-positives
        assert any(
            i.state > HPVInfectionState.INFECTED
            for i in screening_positives
        )
        # assert there are some false-positives
        assert any(
            i.state <= HPVInfectionState.INFECTED
            for i in screening_positives
        )

    def test_simulation_step(self, cohort):
        initial_history = deepcopy(cohort.history)
        initial_length = len(initial_history)
        initial_age = cohort.age
        cohort.next()
        assert len(cohort.history) == initial_length + 1
        assert cohort.age > initial_age

    @pytest.mark.parametrize("iterations", [1, 10, 100])
    def test_prevalences(self, cohort, iterations):
        for _ in range(iterations):
            cohort.next()
        prevalences = cohort.prevalences
        total_prevalence = sum(prevalences.values())
        assert 0 <= total_prevalence <= 1

    @pytest.mark.parametrize("iterations", [20, 100])
    def test_outcomes(self, cohort, iterations):
        for _ in range(iterations):
            cohort.next()
        outcomes = cohort.outcomes
        assert all(
            isinstance(count, int) or isinstance(count, float)
            for outcomes_by_time in outcomes.values()
            for count in outcomes_by_time.values()
        )
        assert all(t >= 0 for t in outcomes.keys())
        assert all(
            isinstance(outcome, ObservableOutcome)
            for outcomes_by_time in outcomes.values()
            for outcome in outcomes_by_time.keys()
        )
        assert any(
            sum(outcomes[t].values()) > 0
            for t in outcomes.keys()
        )

    def test_prevalence_increases_with_incidence(self, cohort):
        cohort_high_incidence = deepcopy(cohort)
        cohort_high_incidence.incidences = {
            k: 10 * v for k, v in cohort.incidences.items()
        }
        for _ in range(50):
            cohort.next()
            cohort_high_incidence.next()
        prevalence_normal = sum(cohort.prevalences.values())
        prevalence_high = sum(cohort_high_incidence.prevalences.values())
        assert prevalence_normal < prevalence_high

    def test_ylls_decrease_with_vaccination(
        self,
        cohort,
        cohort_low_vaccination,
    ):
        for _ in range(300):
            cohort.next()
            cohort_low_vaccination.next()
        ylls_normal = cohort.outcomes_accumulated[ObservableOutcome.YLL]
        ylls_low_vaccination = (
            cohort_low_vaccination.outcomes_accumulated[ObservableOutcome.YLL]
        )
        assert ylls_low_vaccination > ylls_normal

    def test_ylls_decrease_with_screening(
        self,
        cohort,
        cohort_without_screening,
    ):
        for _ in range(300):
            cohort.next()
            cohort_without_screening.next()

        # sanity checks: screenings and colposcopies should be zero for the no
        # screening cohort
        assert (
            cohort_without_screening.outcomes_accumulated[
                ObservableOutcome.SCREENINGS
            ] == 0
        )
        assert (
            cohort_without_screening.outcomes_accumulated[
                ObservableOutcome.COLPOSCOPIES
            ] == 0
        )
        ylls_with_screening = (
            cohort.outcomes_accumulated[ObservableOutcome.YLL]
        )
        ylls_without_screening = (
            cohort_without_screening.outcomes_accumulated[
                ObservableOutcome.YLL
                ]
        )
        assert ylls_with_screening < ylls_without_screening

    def test_ylls_increase_with_followup_loss(self, cohort):
        cohort_high_followup_loss = deepcopy(cohort)
        cohort_high_followup_loss.screening_followup_loss = 0.95
        for _ in range(360):
            cohort.next()
            cohort_high_followup_loss.next()
        # sanity checks: colposcopies should decrease with follow-up loss
        assert (
            cohort.outcomes_accumulated[ObservableOutcome.COLPOSCOPIES] >
            cohort_high_followup_loss.outcomes_accumulated[
                ObservableOutcome.COLPOSCOPIES
            ]
        )

        ylls_normal = cohort.outcomes_accumulated[ObservableOutcome.YLL]
        ylls_high_followup_loss = (
            cohort_high_followup_loss.outcomes_accumulated[
                ObservableOutcome.YLL
            ]
        )
        assert ylls_high_followup_loss > ylls_normal

    def test_cin_detections_increase_with_compliance(
        self,
        cohort,
    ):
        cohort_low_compliance = deepcopy(cohort)
        cohort_low_compliance.screening_compliance = 0.1
        for _ in range(300):
            cohort.next()
            cohort_low_compliance.next()

        # sanity checks: screenings and colposcopies should increase with
        # compliance
        assert (
            cohort.outcomes_accumulated[ObservableOutcome.SCREENINGS] >
            cohort_low_compliance.outcomes_accumulated[
                ObservableOutcome.SCREENINGS
            ]
        )
        assert (
            cohort.outcomes_accumulated[ObservableOutcome.COLPOSCOPIES] >
            cohort_low_compliance.outcomes_accumulated[
                ObservableOutcome.COLPOSCOPIES
            ]
        )

        detections_normal = (
            cohort.outcomes_accumulated[ObservableOutcome.CIN2_DETECTIONS]
            + cohort.outcomes_accumulated[ObservableOutcome.CIN3_DETECTIONS]
        )
        detections_low_compliance = (
            cohort_low_compliance.outcomes_accumulated[
                ObservableOutcome.CIN2_DETECTIONS
            ]
            + cohort_low_compliance.outcomes_accumulated[
                ObservableOutcome.CIN3_DETECTIONS
            ]
        )
        assert detections_normal > detections_low_compliance

    def tests_ylls_decrease_with_compliance(
        self,
        cohort,
    ):
        cohort_low_compliance = deepcopy(cohort)
        cohort_low_compliance.screening_compliance = 0.1
        for _ in range(300):
            cohort.next()
            cohort_low_compliance.next()
        ylls_normal = cohort.outcomes_accumulated[ObservableOutcome.YLL]
        ylls_low_compliance = (
            cohort_low_compliance.outcomes_accumulated[ObservableOutcome.YLL]
        )
        assert ylls_normal < ylls_low_compliance
