import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    from copy import deepcopy

    from hpv_progression_model.common import Snapshot, RNG, HPVInfectionState, ObservableOutcome, CANCER_STATES, UNDETECTED_CANCER_STATES
    from hpv_progression_model.evaluation import compare_differences_in_outcomes, compare_dichotomous_outcomes, SimulationResults
    from hpv_progression_model.model import Cohort
    from hpv_progression_model.params import INCIDENCES, PAP_SMEAR_3YRS_25_64, MAX_FOLLOW_UP_DURATION
    return (
        CANCER_STATES,
        Cohort,
        HPVInfectionState,
        INCIDENCES,
        MAX_FOLLOW_UP_DURATION,
        ObservableOutcome,
        PAP_SMEAR_3YRS_25_64,
        RNG,
        SimulationResults,
        Snapshot,
        UNDETECTED_CANCER_STATES,
        compare_dichotomous_outcomes,
        compare_differences_in_outcomes,
        deepcopy,
    )


@app.cell
def __(PAP_SMEAR_3YRS_25_64):
    intervention_duration_months = 48
    quadrivalent_coverage = 0.0
    screening_regimen = PAP_SMEAR_3YRS_25_64
    screening_compliance = 0.701
    screening_followup_loss = 0.0329
    age_sexual_initiation = 15
    intervention_rate_ratio = 3.1 / 1.3
    procedure_reporting_prob_overall = 0.354
    procedure_reporting_prob_intervention = 0.541
    discount_rate = 0.043
    return (
        age_sexual_initiation,
        discount_rate,
        intervention_duration_months,
        intervention_rate_ratio,
        procedure_reporting_prob_intervention,
        procedure_reporting_prob_overall,
        quadrivalent_coverage,
        screening_compliance,
        screening_followup_loss,
        screening_regimen,
    )


@app.cell
def __(intervention_duration_months, screening_regimen):
    population_structure = {
        20: 1421,
        25: 1459,
        30: 1454,
        35: 1518,
        40: 1487,
        45: 1332,
        50: 1190,
        55: 1107,
        60: 992,
    }
    sample_multiplier = 7  # use this to get a larger sample

    # add population that will enter the target age group during the intervention
    population_group_before_target_age = population_structure.pop(
        int(screening_regimen.start_age - 5),
    )
    intervention_duration_years = int(intervention_duration_months / 12)
    for year in range(
        int(screening_regimen.start_age - intervention_duration_years + 1),
        int(screening_regimen.start_age),
    ):
        population_structure[year] = int(population_group_before_target_age / 5)

    population_structure = {k: int(v * sample_multiplier) for k, v in population_structure.items()}
    return (
        intervention_duration_years,
        population_group_before_target_age,
        population_structure,
        sample_multiplier,
        year,
    )


@app.cell
def __(
    CANCER_STATES,
    Cohort,
    HPVInfectionState,
    MAX_FOLLOW_UP_DURATION,
    ObservableOutcome,
    RNG,
):
    def apply_messaging_intervention(
        cohort: Cohort,
        intervention_duration_months: int,
        intervention_rate_ratio: float,
        procedure_reporting_prob_intervention: float = 1.0,
        procedure_reporting_prob_overall: float = 1.0,
        discount_rate: float = 0.0,
    ) -> tuple[int, float, Cohort]:

        # initialize individuals attributes relative to having received reminders
        # and being with their screening status in the administrative systems
        for individual in cohort.individuals:
            individual.received_message_for_current_screening_cycle = False
            if (
                individual.last_screening_age_months is not None
                and not individual.screening_is_due
                and procedure_reporting_prob_overall >= RNG.random()
            ):
                individual.screening_ok_in_system = True
            else:
                individual.screening_ok_in_system = False

        # start intervention
        messages_sent = 0
        messages_sent_discounted = 0.0
        t_delta = 0
        while (
            cohort.num_individuals_alive > 0
            and cohort.t < MAX_FOLLOW_UP_DURATION
            and t_delta < intervention_duration_months
        ):
            # reset screening status in the administrative system if the last screening
            # was performed more than 3 years ago
            if individual.time_since_last_screening > 36:
                individual.screening_ok_in_system = False

            # send reminders to eligible citizens
            intervention_eligible_citizens = set()
            for individual in cohort.individuals:
                if (
                    individual.age >= 25
                    and individual.age < 65
                    # assume death reporting is perfect
                    and individual.state != HPVInfectionState.DECEASED  
                    and not individual.screening_ok_in_system
                    and not individual.received_message_for_current_screening_cycle
                ):
                    messages_sent += 1
                    messages_sent_discounted += (1 - discount_rate) ** (t_delta / 12)
                    individual.received_message_for_current_screening_cycle = True
                    intervention_eligible_citizens.add(individual)

            # simulate intervention-agnostic progression for this month
            cohort.next()

            # count screenings that would happen this month regardless of the intervention, and
            # report some of them in the administrative system
            num_screenings_regardless_intervention_reported = 0
            for individual in cohort.individuals:
                if (
                    individual.time_since_last_screening == 1
                ):  # screened in this iteration
                    if procedure_reporting_prob_overall >= RNG.random():
                        # report a new screening (independent of the intervention)
                        individual.screening_ok_in_system = True
                        num_screenings_regardless_intervention_reported += 1

            # simulate the additional effect of the intervention
            num_screenings_due_to_intervention_reported = int(
                num_screenings_regardless_intervention_reported
                * (intervention_rate_ratio - 1)
            )
            num_screenings_due_to_intervention = int(
                num_screenings_due_to_intervention_reported
                / procedure_reporting_prob_intervention
            )
            target_individuals_for_intervention = [
                i
                for i in intervention_eligible_citizens
                if i.screening_is_due  # ignore people who got screened anyway
            ]
            # cap number of additional screenings to the size of the pool of eligible citizens
            num_screenings_due_to_intervention = min(
                num_screenings_due_to_intervention,
                len(target_individuals_for_intervention),
            )
            for individual in RNG.choice(
                target_individuals_for_intervention,
                size=num_screenings_due_to_intervention,
            ):
                # manually mimic screening downstream consequences for individuals screened
                # due to the intervention. 
                # NOTE: We use `t - 1` as the time reference because the cohort clock has
                # already advanced one month when we called `cohort.next()` above, but we
                # are still dealing with additional routines caused by the intervention in
                # the previous month
                previous_state = individual.state
                screening_result = individual.screen()
                cohort._outcomes[cohort.t - 1][ObservableOutcome.SCREENINGS] += 1
                if screening_result:
                    if individual.state == HPVInfectionState.CIN2:
                        cohort._outcomes[cohort.t - 1][
                            ObservableOutcome.CIN2_DETECTIONS
                        ] += 1
                    if individual.state == HPVInfectionState.CIN3:
                        cohort._outcomes[cohort.t - 1][
                            ObservableOutcome.CIN3_DETECTIONS
                        ] += 1
                    if (1 - cohort.screening_followup_loss) >= RNG.random():
                        if individual.state == HPVInfectionState.CIN2:
                            cohort._outcomes[cohort.t - 1][ObservableOutcome.EXCISIONS_TYPES_1_2] += 1
                            cohort._outcomes[cohort.t - 1][ObservableOutcome.BIOPSIES] += 1
                        if individual.state == HPVInfectionState.CIN3:
                            cohort._outcomes[cohort.t - 1][ObservableOutcome.EXCISIONS_TYPE_3] += 1
                            cohort._outcomes[cohort.t - 1][ObservableOutcome.BIOPSIES] += 1
                        if individual.state in CANCER_STATES:
                            cohort._outcomes[cohort.t - 1][ObservableOutcome.BIOPSIES] += 1
                        individual.see_and_treat_lesions()

                # record additional transitions after intervention
                if individual.current_state != previous_state:
                    individual._previous_state = previous_state
                    cohort._history_transitions[cohort.t - 1][
                        (
                            individual.previous_state,
                            individual.current_state,
                        )
                    ] += 1
                # record cancer detections from additional screenings due to intervention 
                if (
                    individual.previous_state == HPVInfectionState.LOCAL_UNDETECTED
                    and individual.current_state
                    == HPVInfectionState.LOCAL_DETECTED
                ):
                    cohort._outcomes[cohort.t - 1][
                        ObservableOutcome.LOCAL_DETECTIONS
                    ] += 1
                if (
                    individual.previous_state
                    == HPVInfectionState.REGIONAL_UNDETECTED
                    and individual.current_state
                    == HPVInfectionState.REGIONAL_DETECTED
                ):
                    cohort._outcomes[cohort.t - 1][
                        ObservableOutcome.REGIONAL_DETECTIONS
                    ] += 1
                if (
                    individual.previous_state
                    == HPVInfectionState.DISTANT_UNDETECTED
                    and individual.current_state
                    == HPVInfectionState.DISTANT_DETECTED
                ):
                    cohort._outcomes[cohort.t - 1][
                        ObservableOutcome.DISTANT_DETECTIONS
                    ] += 1

                # record procedure in the administrative system (sometimes)
                if procedure_reporting_prob_intervention >= RNG.random():
                    individual.screening_ok_in_system = True

            # reset reminder message flag for this screening cycle for individuals
            # who are known to have had the screening
            for individual in cohort.individuals:
                if individual.screening_ok_in_system:
                    individual.reminded_of_current_screening_round = False

            t_delta += 1

        return messages_sent, messages_sent_discounted, cohort
    return (apply_messaging_intervention,)


@app.cell
def __(
    Cohort,
    INCIDENCES,
    SimulationResults,
    Snapshot,
    age_sexual_initiation,
    apply_messaging_intervention,
    compare_dichotomous_outcomes,
    compare_differences_in_outcomes,
    deepcopy,
    discount_rate,
    intervention_duration_months,
    intervention_rate_ratio,
    population_structure,
    procedure_reporting_prob_intervention,
    procedure_reporting_prob_overall,
    quadrivalent_coverage,
    screening_compliance,
    screening_followup_loss,
    screening_regimen,
):
    results_by_age = {
        "raw_results_treatment": dict(),
        "raw_results_counterfactual": dict(),
        "messages_sent": dict(),
        "messages_sent_discounted": dict(),
        "effect_metrics": dict(),
        "differences_in_outcomes_discounted": dict(),
    }

    if discount_rate > 0.0:
        comparisons_by_age_discounted = dict()

    for present_age, cohort_size in population_structure.items():
        cohort = Cohort(
            age=age_sexual_initiation,
            num_individuals=cohort_size,
            incidences_by_age=INCIDENCES,
            quadrivalent_coverage=quadrivalent_coverage,
            screening_regimen=screening_regimen,
            screening_compliance=screening_compliance,
            screening_followup_loss=screening_followup_loss,
        )
        while cohort.age <= present_age:
            cohort.next()
        baseline = Snapshot(cohort)
        counterfactual = deepcopy(cohort)

        # apply treatment and monitor follow-up period
        messages_sent, messages_sent_discounted, cohort_after_treatment = apply_messaging_intervention(
            cohort,
            intervention_duration_months=intervention_duration_months,
            intervention_rate_ratio=intervention_rate_ratio,
            procedure_reporting_prob_intervention=procedure_reporting_prob_intervention,
            procedure_reporting_prob_overall=procedure_reporting_prob_overall,
            discount_rate=discount_rate,
        )
        while cohort_after_treatment.num_individuals_alive > 0:
            cohort_after_treatment.next()
        endline_treatment = Snapshot(cohort_after_treatment)

        while counterfactual.num_individuals_alive > 0:
            counterfactual.next()
        endline_counterfactual = Snapshot(counterfactual)

        results_treatment = SimulationResults(
            baseline,
            endline_treatment,
        )
        results_counterfactual = SimulationResults(
            baseline,
            endline_counterfactual,
        )
        results_by_age["raw_results_treatment"][present_age] = results_treatment
        results_by_age["raw_results_counterfactual"][present_age] = results_counterfactual
        results_by_age["messages_sent"][present_age] = messages_sent
        results_by_age["messages_sent_discounted"][present_age] = messages_sent_discounted
        results_by_age["effect_metrics"][present_age] = compare_dichotomous_outcomes(
            results_treatment,
            results_counterfactual,
        )
        results_by_age["differences_in_outcomes_discounted"][present_age] = (
            compare_differences_in_outcomes(
                results_treatment,
                results_counterfactual,
                discount_rate_yearly=discount_rate,
            )
        )
    return (
        baseline,
        cohort,
        cohort_after_treatment,
        cohort_size,
        comparisons_by_age_discounted,
        counterfactual,
        endline_counterfactual,
        endline_treatment,
        messages_sent,
        messages_sent_discounted,
        present_age,
        results_by_age,
        results_counterfactual,
        results_treatment,
    )


@app.cell
def __(results_by_age):
    import pickle
    from pathlib import Path

    with open(Path.cwd() / "messaging_intervention_results_by_age.pickle", "wb") as f:
        f.write(pickle.dumps(results_by_age))
    return Path, f, pickle


@app.cell
def __(results_by_age):
    # messages sent
    messages_sent_total = sum(
        messages_sent
        for messages_sent in results_by_age["messages_sent"].values()
    )
    messages_sent_total
    return (messages_sent_total,)


@app.cell
def __(results_by_age):
    # messages sent
    messages_sent_total_discounted = sum(
        messages_discounted
        for messages_discounted in results_by_age["messages_sent_discounted"].values()
    )
    int(round(messages_sent_total_discounted))
    return (messages_sent_total_discounted,)


@app.cell
def __(population_structure):
    # number of people treated, assuming 100% reach
    # this is an approximation of the number of people treated,
    # as it ignores people who enter the target group during the
    # intervention by turning 25, or leave it by turning 65
    target_group_size = sum(
        pop
        for age, pop in population_structure.items()
        if age >= 25
    )
    target_group_size
    return (target_group_size,)


@app.cell
def __(ObservableOutcome, population_structure, results_by_age):
    differences_in_outcomes_discounted = {
        outcome: sum(
            results_by_age["differences_in_outcomes_discounted"][age][outcome]
            for age in population_structure
        )
        for outcome in ObservableOutcome
    }

    differences_in_outcomes_discounted
    return (differences_in_outcomes_discounted,)


@app.cell
def __(differences_in_outcomes_discounted, target_group_size):
    differences_per_person_discounted = {
        k: v / target_group_size for k, v in differences_in_outcomes_discounted.items()
    }

    differences_per_person_discounted
    return (differences_per_person_discounted,)


@app.cell
def __(ObservableOutcome, results_by_age):
    differences_in_outcomes_undiscounted = {
        outcome: sum(
            metrics[outcome].absolute_difference
            for metrics
            in results_by_age["effect_metrics"].values()
        )
        for outcome in ObservableOutcome
    }

    differences_in_outcomes_undiscounted
    return (differences_in_outcomes_undiscounted,)


@app.cell
def __(ObservableOutcome, population_structure, results_by_age):
    treatment_outcomes = {
        outcome: sum(
            results_by_age["raw_results_treatment"][age].endline.outcomes_accumulated[outcome]
            - results_by_age["raw_results_treatment"][age].baseline.outcomes_accumulated[outcome]
            for age in population_structure.keys()
        )
        for outcome in ObservableOutcome
    }

    treatment_outcomes
    return (treatment_outcomes,)


@app.cell
def __(ObservableOutcome, population_structure, results_by_age):
    counterfactual_outcomes = {
        outcome: sum(
            results_by_age["raw_results_counterfactual"][age].endline.outcomes_accumulated[outcome]
            - results_by_age["raw_results_counterfactual"][age].baseline.outcomes_accumulated[outcome]
            for age in population_structure.keys()
        )
        for outcome in ObservableOutcome
    }

    counterfactual_outcomes
    return (counterfactual_outcomes,)


@app.cell
def __(ObservableOutcome, population_structure, results_by_age):
    import numpy as np

    risk_ratios = {}
    for outcome_ in ObservableOutcome:
        intervention_events = sum(
            results_by_age["raw_results_treatment"][age].events[outcome_]
            for age
            in population_structure.keys()
        )
        intervention_sample = intervention_events + sum(
            results_by_age["raw_results_counterfactual"][age].no_events[outcome_]
            for age
            in population_structure.keys()
        )
        comparator_events = sum(
            results_by_age["raw_results_counterfactual"][age].events[outcome_]
            for age
            in population_structure.keys()
        )
        comparator_sample = comparator_events + sum(
            results_by_age["raw_results_counterfactual"][age].no_events[outcome_]
            for age
            in population_structure.keys()
        )
        risk_ratios[outcome_] = (
            (intervention_events / intervention_sample)
            / (comparator_events / comparator_sample)
        )

    risk_ratios
    return (
        comparator_events,
        comparator_sample,
        intervention_events,
        intervention_sample,
        np,
        outcome_,
        risk_ratios,
    )


if __name__ == "__main__":
    app.run()
