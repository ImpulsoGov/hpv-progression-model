# -*- coding: utf-8 -*-

"""Common functionality for the HPV progression model.

This module provides shared utilities and base classes for the `hpv_progression_model` package. It defines enumerations for HPV genotypes, infection states, and observable outcomes. The module also includes abstract base classes for immutable objects and those representing longitudinal time series data. 

Additionally, it defines constants and attributes commonly used throughout the package, such as random number generation and a tuple for categorizing
undetected cancer stages.

Attributes:
    RNG (np.random.Generator): A random number generator initialized with a fixed seed (42) for deterministic behavior in simulations.
    
    UNDETECTED_CANCER_STATES (tuple[HPVInfectionState]): A tuple containing HPV infection states representing undetected local, regional, and distant
    cancer stages.

    FIGO_TO_LRD_STAGES (dict[CervicalCancerFIGOStage, HPVInfectionState]): A dictionary mapping FIGO stages to LRD stages.

Example:
    Example usage of the module-level attributes:

    Using the `RNG` for generating random numbers in a simulation:

    ```py
    >>> from hpv_progression_model.common import RNG
    >>> RNG.random()
    0.7739560485559633
    ```

    Accessing the undetected cancer stages:

    ```py
    >>> from hpv_progression_model.common import UNDETECTED_CANCER_STATES
    >>> UNDETECTED_CANCER_STATES
    (<HPVInfectionState.LOCAL_UNDETECTED: 4>, 
        <HPVInfectionState.REGIONAL_UNDETECTED: 6>, 
        <HPVInfectionState.DISTANT_UNDETECTED: 8>)
    ```

    Converting FIGO stages to Local-Regional-Distant stages:

    ```py
    >>> from hpv_progression_model.common import FIGO_TO_LRD_STAGES
    >>> FIGO_TO_LRD_STAGES[FIGO_TO_LRD_STAGES.I]
    <HPVInfectionState.LOCAL_DETECTED: 5>
    ```
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import cast, Generic, TypeVar

import numpy as np


T = TypeVar("T")

# Random number generator for consistent simulation results
RNG = np.random.default_rng(seed=42)


class Immutable(ABC):
    """Abstract base class that prevents setting or deleting attributes after initialization.

    This class can be inherited to create immutable objects, ensuring
    that once attributes are set, they cannot be changed or deleted.

    Raises:
        TypeError: If trying to set or delete an attribute.
    """

    def __setattr__(self, *args):
        """Prevents setting attributes."""
        raise TypeError(
            f"Setting attributes is not allowed for `{self.__class__}` objects",
        )

    def __delattr__(self, *args):
        """Prevents deleting attributes."""
        raise TypeError(
            f"Deleting attributes is not allowed for `{self.__class__}` objects",
        )


class Longitudinal(ABC):
    """Abstract base class to model objects that evolve over time.

    This class defines a time (`_t`) attribute and provides an interface for
    incrementing time in simulations that involve longitudinal progression.
    """

    @abstractmethod
    def __init__(self):
        """Initializes the starting time step."""
        self._t = 0

    @property
    def t(self) -> int:
        """Returns the current time step in months.

        Returns:
            int: The current time step in months.
        """
        return self._t

    @abstractmethod
    def next(self) -> None:
        """Advances the time by one month."""
        self._t += 1


class CervicalCancerFIGOStage(IntEnum):
    """Enumeration of the FIGO staging of cervical cancer. 

    A simplified version of the 2018 revision of the International Federation of Gynaecology and Obstetrics (FIGO) staging of cervical cancer, going
    from stage I to stage IV.

    Attributes:
        I: FIGO stage I.
        II: FIGO stage II.
        III: FIGO stage III.
        IV: FIGO stage IV.
    """
    I = 1
    II = 2
    III = 3
    IV = 4


class HPVGenotype(StrEnum):
    """Enumeration of HPV genotypes relevant to the model.

    Attributes:
        HPV_16: HPV Type 16
        HPV_18: HPV Type 18
        HPV_31: HPV Type 31
        HPV_33: HPV Type 33
        HPV_45: HPV Type 45
        HPV_52: HPV Type 52
        HPV_55: HPV Type 55
        OTHER_HR: Other high-risk HPV types
        OTHER_LR: Other low-risk HPV types
    """
    HPV_16 = "HPV 16"
    HPV_18 = "HPV 18"
    HPV_31 = "HPV 31"
    HPV_33 = "HPV 33"
    HPV_45 = "HPV 45"
    HPV_52 = "HPV 52"
    HPV_58 = "HPV 58"
    OTHER_HR = "Other, High-Risk"
    OTHER_LR = "Other, Low-Risk"


class HPVInfectionState(IntEnum):
    """Enumeration of states representing the progression of HPV infection and cancer.

    The states range from healthy to various cancer stages, both detected and undetected.

    Attributes:
        HEALTHY: Individual without HPV infection.
        INFECTED: Active HPV infection, lesion-free or CIN 1/ASCUS.
        CIN2: Cervical intraepithelial neoplasia (CIN 2) stage.
        CIN3: Cervical intraepithelial neoplasia (CIN 3) stage.
        LOCAL_UNDETECTED: Localized cancer, undetected.
        LOCAL_DETECTED: Localized cancer, detected.
        REGIONAL_UNDETECTED: Regional cancer, undetected.
        REGIONAL_DETECTED: Regional cancer, detected.
        DISTANT_UNDETECTED: Distant metastasis, undetected.
        DISTANT_DETECTED: Distant metastasis, detected.
        DECEASED: Individual has passed away.
    """
    HEALTHY = 0
    INFECTED = 1
    CIN2 = 2
    CIN3 = 3
    LOCAL_UNDETECTED = 4
    LOCAL_DETECTED = 5
    REGIONAL_UNDETECTED = 6
    REGIONAL_DETECTED = 7
    DISTANT_UNDETECTED = 8
    DISTANT_DETECTED = 9
    DECEASED = 10


class ObservableOutcome(StrEnum):
    """Enumeration of observable health outcomes relevant to the model.

    Attributes:
        SCREENINGS: Number of screenings.
        COLPOSCOPIES: Number of colposcopy procedures.
        CIN2_DETECTIONS: Number of CIN 2 lesions detected.
        CIN3_DETECTIONS: Number of CIN 3 lesions detected.
        LOCAL_DETECTIONS: Number of invasive cancers detected at the local stage.
        REGIONAL_DETECTIONS: Number of invasive cancers detected at the regional stage.
        DISTANT_DETECTIONS: Number of invasive cancers detected at the distant stage.
        DEATHS: Number of deaths recorded.
    """
    SCREENINGS = "Cervical cancer screenings"
    COLPOSCOPIES = "Colposcopies"
    CIN2_DETECTIONS = "CIN 2 detections"
    CIN3_DETECTIONS = "CIN 3 detections"
    EXCISIONS_TYPES_1_2 = "Excisions of types 1-2"
    EXCISIONS_TYPE_3 = "Excisions of type 3"
    BIOPSIES = "Biopsies"
    LOCAL_DETECTIONS = "Invasive cancer detections - Local stage"
    REGIONAL_DETECTIONS = "Invasive cancer detections - Regional stage"
    DISTANT_DETECTIONS = "Invasive cancer detections - Distant stage"
    DEATHS = "deaths"
    YLL = "Years of Life Lost"


@dataclass
class ScreeningMethod:
    """Represents a method of screening for HPV-related diseases.

    Attributes:
        sensitivity (dict[HPVInfectionState, float]): Sensitivity of the screening method for each HPV infection state.
        specificity (float): Specificity of the screening method (applies to all states).
    """
    sensitivity: dict[HPVInfectionState, float]
    specificity: float


class ScreeningRegimen:
    """Represents a set of rules determining screening eligibility and method.

    A screening regimen applies a series of rules to decide which screening
    method is recommended for an individual based on their age, time since the
    last screening, and previous screening results.

    Args:
        rule (Callable[[float, bool | None], tuple[ScreeningMethod, float]): A
        function that receives the age and last screening result and returns a
        tuple of method and the prescribed interval from the last screening,
        in months.
        start_age (int | float, optional): The age at which the screening
        should start according to the regimen, in years.
        name (str): Name of the screening regimen.
        description (str | None, optional): An optional long description of
        the regimen.
    """

    def __init__(
        self,
        rule: Callable[[float, bool], tuple[ScreeningMethod, int]],
        start_age: int | float,
        name: str,
        description: str | None = None,
    ):
        self.rule = rule
        self.name = name
        self.start_age = float(start_age)
        self.description = description

    def get_recommendation(
        self,
        age: int | float,
        last_screening_result: bool,
    ) -> tuple[ScreeningMethod, int]:
        """Determines when to perform a screening.
        
        Args:
            age (int | float): The age of the individual.
            last_screening_result (bool): The result of the last screening.
        
        Returns:
            tuple[ScreeningMethod, int]: The recommended screening method and number of months since the last screening.
        """
        screening_method, screening_interval = self.rule(
            age=age,
            last_screening_result=last_screening_result,
        )
        return screening_method, screening_interval


class Snapshot(Generic[T], Immutable):
    """An read-only copy of an object's state"""

    def __new__(cls, obj):
        obj_copy = deepcopy(obj)
        obj_copy.__setattr__ = super().__setattr__
        obj_copy.__delattr__ = super().__delattr__
        return cast(Snapshot, obj_copy)


CANCER_STATES: tuple[HPVInfectionState] = (
    HPVInfectionState.LOCAL_UNDETECTED,
    HPVInfectionState.LOCAL_DETECTED,
    HPVInfectionState.REGIONAL_UNDETECTED,
    HPVInfectionState.REGIONAL_DETECTED,
    HPVInfectionState.DISTANT_UNDETECTED,
    HPVInfectionState.DISTANT_DETECTED,
)

# Cancer stages where the disease is undetected
UNDETECTED_CANCER_STATES: tuple[HPVInfectionState] = (
    HPVInfectionState.LOCAL_UNDETECTED,
    HPVInfectionState.REGIONAL_UNDETECTED,
    HPVInfectionState.DISTANT_UNDETECTED,
)

FIGO_TO_LRD_STAGES: dict[CervicalCancerFIGOStage, HPVInfectionState] = {
    CervicalCancerFIGOStage.I: HPVInfectionState.LOCAL_DETECTED,
    CervicalCancerFIGOStage.II: HPVInfectionState.REGIONAL_DETECTED,
    CervicalCancerFIGOStage.III: HPVInfectionState.REGIONAL_DETECTED,
    CervicalCancerFIGOStage.IV: HPVInfectionState.DISTANT_DETECTED,
}
