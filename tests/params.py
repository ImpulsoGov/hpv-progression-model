from numpy.testing import assert_almost_equal

from hpv_progression_model.common import HPVInfectionState
from hpv_progression_model.params import (
    CONSIDER_DETECTED_CANCERS_CURED_AT,
    MAX_FOLLOW_UP_DURATION,
    TRANSITION_PROBABILITIES
)

def test_transition_probabilities():
    for transition_probabilities in TRANSITION_PROBABILITIES.values():
        assert transition_probabilities.shape == (
            MAX_FOLLOW_UP_DURATION,
            len(HPVInfectionState),
            len(HPVInfectionState)
        )
        for t, monthly_probabilities in enumerate(transition_probabilities):
            for state in HPVInfectionState:
                assert_almost_equal(sum(monthly_probabilities[state]), 1.0)
                assert all(
                    prob >= 0.0 for prob in monthly_probabilities[state]
                )
            
            # assert detecting a cancer is not net negative for mortality
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_UNDETECTED,
                HPVInfectionState.DECEASED
            ] >= monthly_probabilities[
                HPVInfectionState.LOCAL_DETECTED,
                HPVInfectionState.DECEASED
            ]
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_UNDETECTED,
                HPVInfectionState.DECEASED
            ] >= monthly_probabilities[
                HPVInfectionState.REGIONAL_DETECTED,
                HPVInfectionState.DECEASED
            ]
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_UNDETECTED,
                HPVInfectionState.DECEASED
            ] >= monthly_probabilities[
                HPVInfectionState.DISTANT_DETECTED,
                HPVInfectionState.DECEASED
            ]

            if t < CONSIDER_DETECTED_CANCERS_CURED_AT:
                # assert mortality worsens with the cancer stage
                assert monthly_probabilities[
                    HPVInfectionState.REGIONAL_UNDETECTED,
                    HPVInfectionState.DECEASED
                ] > monthly_probabilities[
                    HPVInfectionState.LOCAL_UNDETECTED,
                    HPVInfectionState.DECEASED
                ]
                assert monthly_probabilities[
                    HPVInfectionState.DISTANT_UNDETECTED,
                    HPVInfectionState.DECEASED
                ] > monthly_probabilities[
                    HPVInfectionState.REGIONAL_UNDETECTED,
                    HPVInfectionState.DECEASED
                ]
                assert monthly_probabilities[
                    HPVInfectionState.REGIONAL_DETECTED,
                    HPVInfectionState.DECEASED
                ] > monthly_probabilities[
                    HPVInfectionState.LOCAL_DETECTED,
                    HPVInfectionState.DECEASED
                ]
                assert monthly_probabilities[
                    HPVInfectionState.DISTANT_DETECTED,
                    HPVInfectionState.DECEASED
                ] > monthly_probabilities[
                    HPVInfectionState.REGIONAL_DETECTED,
                    HPVInfectionState.DECEASED
                ]
            else:
                # assert extra mortality from cancer is zero after cure
                # assert mortality worsens with the cancer stage
                assert monthly_probabilities[
                    HPVInfectionState.LOCAL_DETECTED,
                    HPVInfectionState.DECEASED
                ] == 0.0
                assert monthly_probabilities[
                    HPVInfectionState.REGIONAL_DETECTED,
                    HPVInfectionState.DECEASED
                ] == 0.0
                assert monthly_probabilities[
                    HPVInfectionState.DISTANT_DETECTED,
                    HPVInfectionState.DECEASED
                ] == 0.0
                

            # assert CIN3 is worse than CIN2
            assert monthly_probabilities[
                HPVInfectionState.CIN3,
                HPVInfectionState.LOCAL_UNDETECTED
            ] >= monthly_probabilities[
                HPVInfectionState.CIN2,
                HPVInfectionState.LOCAL_UNDETECTED
            ]

            # assert some transitions are not possible
            assert monthly_probabilities[
                HPVInfectionState.HEALTHY,
                HPVInfectionState.INFECTED,
            ] == 0.0  # This should happen through exposure only
            assert monthly_probabilities[
                HPVInfectionState.CIN2,
                HPVInfectionState.CIN3,
            ] == 0.0  # This hasn't been implemented in this model
            assert monthly_probabilities[
                HPVInfectionState.CIN3,
                HPVInfectionState.CIN2,
            ] == 0.0  # This hasn't been implemented in this model
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_UNDETECTED,
                HPVInfectionState.DISTANT_UNDETECTED,
            ] == 0.0  # Should pass through the regional stage
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_UNDETECTED,
                HPVInfectionState.HEALTHY,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_UNDETECTED,
                HPVInfectionState.INFECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_UNDETECTED,
                HPVInfectionState.CIN2,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_UNDETECTED,
                HPVInfectionState.CIN3,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_UNDETECTED,
                HPVInfectionState.HEALTHY,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_UNDETECTED,
                HPVInfectionState.INFECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_UNDETECTED,
                HPVInfectionState.CIN2,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_UNDETECTED,
                HPVInfectionState.CIN3,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_UNDETECTED,
                HPVInfectionState.LOCAL_UNDETECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_UNDETECTED,
                HPVInfectionState.HEALTHY,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_UNDETECTED,
                HPVInfectionState.INFECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_UNDETECTED,
                HPVInfectionState.CIN2,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_UNDETECTED,
                HPVInfectionState.CIN3,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_UNDETECTED,
                HPVInfectionState.LOCAL_UNDETECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_UNDETECTED,
                HPVInfectionState.REGIONAL_UNDETECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_DETECTED,
                HPVInfectionState.INFECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_DETECTED,
                HPVInfectionState.CIN2,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.LOCAL_DETECTED,
                HPVInfectionState.CIN3,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_DETECTED,
                HPVInfectionState.INFECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_DETECTED,
                HPVInfectionState.CIN2,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_DETECTED,
                HPVInfectionState.CIN3,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.REGIONAL_DETECTED,
                HPVInfectionState.LOCAL_DETECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_DETECTED,
                HPVInfectionState.INFECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_DETECTED,
                HPVInfectionState.CIN2,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_DETECTED,
                HPVInfectionState.CIN3,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_DETECTED,
                HPVInfectionState.LOCAL_DETECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
            assert monthly_probabilities[
                HPVInfectionState.DISTANT_DETECTED,
                HPVInfectionState.REGIONAL_DETECTED,
            ] == 0.0  # Cancer regressions haven't been implemented
