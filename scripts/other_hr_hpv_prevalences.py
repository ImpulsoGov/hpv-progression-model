# Given data
prevalences = {
    "Type 16": 0.089,
    "Type 18": 0.043,
    "Type 31": 0.048,
    "Type 33": 0.019,
    "Type 45": 0.036,
    "Type 52": 0.088,
    "Type 58": 0.061
}

# Total prevalence of infection
total_prevalence = 0.386

from itertools import combinations
import numpy as np

def calculate_inclusion_exclusion(prevalences):
    # Inclusion-exclusion principle
    sum_single = sum(prevalences.values())
    
    # Calculate pairwise intersections
    sum_pairs = sum(
        prevalences[a] * prevalences[b]
        for a, b in combinations(prevalences.keys(), 2)
    )
    
    # Calculate triple intersections
    sum_triples = sum(
        prevalences[a] * prevalences[b] * prevalences[c]
        for a, b, c in combinations(prevalences.keys(), 3)
    )
    
    # Calculate higher-order intersections if needed
    sum_quadruples = sum(
        prevalences[a] * prevalences[b] * prevalences[c] * prevalences[d]
        for a, b, c, d in combinations(prevalences.keys(), 4)
    )
    
    # Calculate higher-order intersections if needed
    sum_quintuples = sum(
        prevalences[a] * prevalences[b] * prevalences[c] * prevalences[d] * prevalences[e]
        for a, b, c, d, e in combinations(prevalences.keys(), 5)
    )
    
    # Calculate higher-order intersections if needed
    sum_sixtuples = sum(
        prevalences[a] * prevalences[b] * prevalences[c] * prevalences[d] * prevalences[e] * prevalences[f]
        for a, b, c, d, e, f in combinations(prevalences.keys(), 6)
    )

    sum_seventuples = np.prod(list(prevalences.values()))
    
    # Continue this pattern for more intersections
    # Calculate inclusion-exclusion total
    inclusion_exclusion_result = (
        sum_single - sum_pairs + sum_triples - sum_quadruples + sum_quintuples - sum_sixtuples + sum_seventuples
    )
    
    return inclusion_exclusion_result

if __name__ == "__main__":
    # Calculate the probability of being infected with at least one of the listed types
    at_least_one_type = calculate_inclusion_exclusion(prevalences)

    # Calculate the prevalence of other types
    prevalence_other_types = total_prevalence - at_least_one_type

    # Output results
    print(f"Probability of being infected with at least one listed type: {at_least_one_type:.4f}")
    print(f"Prevalence of other types: {prevalence_other_types:.4f}")
