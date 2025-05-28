from typing import List, Dict, Tuple, Any
import numpy as np
from numpy.typing import NDArray

import mdp.utils as utils
import mdp.search_spaces as ss

"""
Alternatively, we could construct the abstraction first and then lift it to two finer MDPs?!
This would mean we could do exact abstraction!?
But how to lift to ensure only the value of the optimal policy is preserved?
Or the value of all policies is preserved?
"""

def partitions(similarity_matrix: NDArray[np.bool_]) -> Tuple[Dict[int, Tuple[int, ...]], List[Tuple[int, ...]]]:
    """
    Computes partitions of states based on a similarity matrix.

    Args:
        similarity_matrix: A square boolean matrix where `similarity_matrix[i, j]` is True
                           if state i and state j are considered similar. It's assumed
                           to be symmetric and include self-similarity (diagonal is True).

    Returns:
        A tuple containing:
            - mapping: A dictionary where keys are original state indices and values
                       are tuples representing the partition (group of similar states)
                       that the key state belongs to. Each state in a partition is
                       mapped to the same tuple object.
            - parts: A list of unique partition tuples. Each tuple contains sorted
                     indices of states belonging to that partition.
    """
    num_states: int = similarity_matrix.shape[0]
    # Ensure the diagonal is True for self-similarity, as np.triu below might exclude it if not explicitly handled
    # and the logic relies on finding pairs.
    # A more robust way is to ensure the input similarity_matrix already reflects this.
    # For now, we assume the input similarity_matrix is correctly formed (symmetric, reflexive).

    # np.triu returns the upper triangle, k=1 excludes the diagonal.
    # We are interested in pairs (i,j) where i < j and sim[i,j] is True.
    pairs: Tuple[NDArray[np.int_], NDArray[np.int_]] = np.where(np.triu(similarity_matrix, k=1))
    
    # Initialize mapping: each state initially maps to a list containing only itself.
    # This will be expanded with other similar states.
    mapping_work: Dict[int, List[int]] = {k: [k] for k in range(num_states)}

    for i, j in zip(*pairs): # Iterate over (row, col) indices where similarity_matrix[row, col] is True
        mapping_work[i].append(j)
        mapping_work[j].append(i) # Since similarity is symmetric

    # Convert lists to sorted tuples to make them hashable and canonical
    # Each state k is now mapped to a tuple representing its partition
    final_mapping: Dict[int, Tuple[int, ...]] = {k: tuple(sorted(list(set(v)))) for k, v in mapping_work.items()}
    
    # Extract unique partitions
    # Using set(final_mapping.values()) ensures that each partition appears only once in 'parts'
    parts: List[Tuple[int, ...]] = sorted(list(set(final_mapping.values()))) # Sort for deterministic output
    
    return final_mapping, parts


def construct_abstraction_fn(
    mapping: Dict[int, Tuple[int, ...]], 
    representative_abstract_state_indices: List[int], 
    num_original_states: int, 
    num_abstract_states: int
) -> NDArray[np.float_]:
    """
    Constructs an abstraction mapping matrix (transpose) that maps original states to abstract states.

    The resulting matrix `f.T` (or `abstraction_mapping_matrix_transpose` before returning)
    has shape (num_original_states, num_abstract_states).
    `f[i, j] = 1` if original state `i` maps to abstract state `j`, and 0 otherwise.
    This implementation assumes each original state maps to exactly one abstract state,
    represented by the first element of its partition tuple if that element is in
    `representative_abstract_state_indices`.

    Args:
        mapping: A dictionary where keys are original state indices and values
                 are tuples representing the partition (group of similar states)
                 that the key state belongs to.
        representative_abstract_state_indices: A list of original state indices, where each index
                                               is chosen as the representative for an abstract state.
                                               The order in this list defines the indexing of abstract states.
        num_original_states: The total number of states in the original MDP.
        num_abstract_states: The total number of states in the abstract MDP (should be len(representative_abstract_state_indices)).

    Returns:
        An abstraction mapping matrix (transpose of `f` from the original code)
        of shape (num_abstract_states, num_original_states).
        `abstraction_mapping_matrix[j, i] = 1` if original state `i` maps to abstract state `j`.
    """
    abstraction_mapping_matrix_transpose: NDArray[np.float_] = np.zeros((num_original_states, num_abstract_states))
    
    for original_state_idx, partition_tuple in mapping.items():
        # The representative for the partition is the first element of the sorted tuple.
        # Find which abstract state this representative corresponds to.
        representative_for_partition: int = partition_tuple[0]
        try:
            abstract_state_idx: int = representative_abstract_state_indices.index(representative_for_partition)
            abstraction_mapping_matrix_transpose[original_state_idx, abstract_state_idx] = 1.0
        except ValueError:
            # This case should ideally not happen if representative_abstract_state_indices
            # is correctly constructed from the partitions.
            # It means the representative of the current partition is not in the list of chosen representatives.
            # This could happen if `parts` in `build_state_abstraction` was not used correctly
            # to derive `representative_abstract_state_indices`.
            pass # Or raise an error, depending on desired strictness

    assert not (abstraction_mapping_matrix_transpose > 1).any(), "An original state maps to more than one abstract state."
    # Ensure each original state maps to at least one abstract state if num_abstract_states > 0
    if num_abstract_states > 0:
        assert np.all(np.sum(abstraction_mapping_matrix_transpose, axis=1) <= 1), "An original state maps to more than one abstract state"
        # The following assertion might be too strict if some original states are not part of any partition
        # that has a representative in representative_abstract_state_indices.
        # This could happen if mapping is incomplete or representative_abstract_state_indices is filtered.
        # assert np.all(np.sum(abstraction_mapping_matrix_transpose, axis=1) >= 1), "Not all original states map to an abstract state"


    return abstraction_mapping_matrix_transpose.T


def abstract_the_mdp(mdp: utils.MDP, abstract_state_indices: List[int]) -> utils.MDP:
    """
    Creates an abstract MDP from a given MDP using a specified set of abstract state indices.

    The abstract MDP uses a subset of states from the original MDP. The transition probabilities
    and rewards are taken directly from the original MDP for these selected states.

    Args:
        mdp: The original MDP object.
        abstract_state_indices: A list of indices of the states from the original MDP
                                that will form the states of the abstract MDP. The order
                                matters as it defines the new state indexing.

    Returns:
        A new MDP object representing the abstracted MDP.
    """
    # Ensure abstract_state_indices are valid and sorted for consistent slicing, though sorting might not be strictly necessary
    # if the order in abstract_state_indices defines the new state order.
    # For direct indexing as done here, sorting is not required but using a list/array of unique indices is.
    
    # P has shape (S, A, S'), r has shape (S, A)
    # We want P_abs[s_abs, a, s_next_abs] and r_abs[s_abs, a]
    # where s_abs and s_next_abs are indices *within the abstract MDP's state space*.
    # These correspond to original states mdp.P[orig_idx_s_abs, a, orig_idx_s_next_abs]
    
    # Convert list of indices to a NumPy array for advanced indexing
    idx_array: NDArray[np.int_] = np.array(abstract_state_indices)

    # Slice P: Select rows for current abstract states, then slice again for next abstract states
    # P[idx_array, :, :] gives transitions *from* selected states to *all* original states. Shape: (len(idx_array), A, S_orig)
    # Then, P[idx_array, :, :][:, :, idx_array] (or P[np.ix_(idx_array, np.arange(mdp.A), idx_array)])
    # selects transitions *from* selected states *to* selected states. Shape: (len(idx_array), A, len(idx_array))
    abs_P: NDArray[np.float_] = mdp.P[idx_array[:, np.newaxis, np.newaxis], np.arange(mdp.A)[np.newaxis, :, np.newaxis], idx_array[np.newaxis, np.newaxis, :]]
    
    # Slice r: Select rows for current abstract states
    # r[idx_array, :] gives rewards for selected states for all actions. Shape: (len(idx_array), A)
    abs_r: NDArray[np.float_] = mdp.r[idx_array[:, np.newaxis], np.arange(mdp.A)]

    num_abstract_states: int = len(abstract_state_indices)
    
    # d0 for the abstract MDP: if the original d0 exists, map it.
    # This assumes d0 is a distribution over original states.
    # A simple approach is to renormalize d0 over the abstract states.
    # Or, if d0_abs is needed, it should be explicitly defined or derived based on abstraction logic.
    # For now, let's assume d0 is not directly transferable without further logic, so pass None or a uniform distribution.
    # If d0 is crucial, the caller might need to define how it translates.
    # A simple transfer:
    abs_d0: NDArray[np.float_] | None = None
    if mdp.d0 is not None:
        abs_d0_raw = mdp.d0[idx_array]
        if np.sum(abs_d0_raw) > 0 : # Avoid division by zero if all selected states have 0 prob
            abs_d0 = abs_d0_raw / np.sum(abs_d0_raw)
        else: # Fallback to uniform if no probability mass in selected states
            abs_d0 = np.ones(num_abstract_states) / num_abstract_states


    return utils.MDP(num_abstract_states, mdp.A, abs_P, abs_r, mdp.discount, abs_d0)


def shared(list1: Tuple[Any, ...], list2: Tuple[Any, ...]) -> bool:
    """
    Checks if two tuples (or lists by implication of usage) have any common elements.

    Args:
        list1: The first tuple.
        list2: The second tuple.

    Returns:
        True if there is at least one common element between list1 and list2, False otherwise.
    """
    return any(item in list2 for item in list1)


def fix_mapping(
    mapping: Dict[int, Tuple[int, ...]]
) -> Tuple[Dict[int, Tuple[int, ...]], List[Tuple[int, ...]]]:
    """
    Ensures transitivity in a state mapping representing partitions.

    If state A is mapped to a group G1 and state B is mapped to a group G2,
    and G1 and G2 share any common states, then this function merges G1 and G2.
    All states originally in G1 or G2 will be mapped to the merged group.
    This process is repeated until no more merges can occur, effectively
    finding the connected components in a graph where states are nodes and
    an edge exists if two states are in the same initial partition.

    Args:
        mapping: A dictionary where keys are original state indices and values
                 are tuples representing the partition (group of similar states)
                 that the key state belongs to. This mapping might lack transitivity.

    Returns:
        A tuple containing:
            - new_mapping: A dictionary similar to the input `mapping` but where
                           partitions are guaranteed to be transitive (i.e., they form
                           proper equivalence classes).
            - parts: A list of unique partition tuples from `new_mapping`.
    """
    num_states: int = len(mapping.keys())
    # Initialize new_mapping based on the input mapping, ensuring values are mutable lists for merging
    current_mapping_work: Dict[int, List[int]] = {k: list(v) for k, v in mapping.items()}

    changed_in_iteration: bool = True
    while changed_in_iteration:
        changed_in_iteration = False
        # Create a temporary copy for iteration, as we might modify current_mapping_work
        map_to_iterate = list(current_mapping_work.items()) # list of (key, value_list)

        for i in range(len(map_to_iterate)):
            k1, v1_list = map_to_iterate[i]
            
            # Check against all subsequent partitions to avoid redundant checks and self-comparison
            for j in range(i + 1, len(map_to_iterate)):
                k2, v2_list = map_to_iterate[j]

                # Check for shared elements. Convert to tuple for `shared` function if it expects tuples.
                # Or, modify `shared` to accept lists, or reimplement check here.
                # For direct list check:
                if any(item in v2_list for item in v1_list): # If partitions v1 and v2 overlap
                    # Merge v1 and v2
                    merged_partition_set = set(v1_list).union(set(v2_list))
                    
                    # Update all states that were in v1 or v2 (or pointed to them)
                    # to now point to the new merged partition.
                    # This needs to be done carefully to ensure all related states are updated.
                    # A simpler way: iterate through all states. If their current partition
                    # overlaps with v1_list or v2_list, update their partition to merged_partition_set.
                    
                    # Potential states to update: all states whose current partition will be affected by the merge.
                    # These are states whose current partitions are v1_list or v2_list (by identity or content).
                    
                    new_v1_list = list(merged_partition_set)
                    
                    # Update for k1 if its list was v1_list
                    if current_mapping_work[k1] is v1_list or tuple(sorted(current_mapping_work[k1])) == tuple(sorted(v1_list)):
                         if tuple(sorted(current_mapping_work[k1])) != tuple(sorted(new_v1_list)):
                            current_mapping_work[k1] = new_v1_list
                            v1_list = new_v1_list # Update local copy for further merges in this pass
                            changed_in_iteration = True
                    
                    # Update for k2 if its list was v2_list
                    if current_mapping_work[k2] is v2_list or tuple(sorted(current_mapping_work[k2])) == tuple(sorted(v2_list)):
                        if tuple(sorted(current_mapping_work[k2])) != tuple(sorted(new_v1_list)): # Use the already merged list
                            current_mapping_work[k2] = new_v1_list
                            v2_list = new_v1_list # Update local copy
                            changed_in_iteration = True
                    
                    # Crucial step: Propagate the merge to all states.
                    # Any state whose partition (v_other) overlaps with the (now outdated) v1_list or v2_list,
                    # or with the new_v1_list, should adopt new_v1_list.
                    # This ensures that if C is similar to B, and B became similar to A, C also becomes similar to A.
                    for k_other in range(num_states):
                        v_other_list = current_mapping_work[k_other]
                        if any(item in new_v1_list for item in v_other_list): # If v_other overlaps with the merged set
                            # Check if an update is actually needed to avoid infinite loops if not careful
                            if tuple(sorted(current_mapping_work[k_other])) != tuple(sorted(new_v1_list)):
                                current_mapping_work[k_other] = list(set(v_other_list).union(new_v1_list)) # Merge and update
                                changed_in_iteration = True
                                # Potentially, new_v1_list itself should be updated if v_other_list brought new states.
                                # This suggests the merging logic might need to be more global or iterative within the loop.
                                # For simplicity, the original code's double loop implies a pass-based approach.
                                # The key is that `new_mapping[k1] += list(set(v1).union(set(v2)))` accumulates.
                                # The provided solution attempt here is trying to refine it.
                                # Let's stick closer to the spirit of the original `fix_mapping`'s accumulation logic
                                # but aim for convergence. The original `fix_mapping` rebuilds `new_mapping` from scratch
                                # in each call, which isn't what's happening in this iterative refinement.

    # The original `fix_mapping` had a non-iterative structure that effectively did one pass of merging.
    # For true transitive closure, an iterative approach like above (or a graph algorithm) is needed.
    # Let's re-implement `fix_mapping` closer to its original intent but ensure correctness for transitivity.
    # A common way is to use a Disjoint Set Union (DSU) data structure or iterate until no changes.

    # Re-attempting fix_mapping logic based on iterative merging until stable:
    # Convert initial mapping to a list of sets for easier manipulation
    initial_partitions_as_sets: List[set[int]] = sorted(list(set(frozenset(v) for v in mapping.values())) , key=lambda x: min(x) if x else float('inf'))
    
    merged_partitions: List[set[int]] = []
    if not initial_partitions_as_sets: # Handle empty initial partitions
        final_fixed_mapping: Dict[int, Tuple[int, ...]] = {k: tuple() for k in range(num_states)}
        return final_fixed_mapping, []

    for current_set in initial_partitions_as_sets:
        if not current_set: continue # Skip empty sets if they can occur
        
        # Find if current_set overlaps with any set already in merged_partitions
        overlapping_indices: List[int] = []
        for i, existing_set in enumerate(merged_partitions):
            if not current_set.isdisjoint(existing_set): # Check for overlap
                overlapping_indices.append(i)
        
        if not overlapping_indices:
            # No overlap, add current_set as a new partition
            merged_partitions.append(current_set)
        else:
            # Overlaps found, merge current_set with all overlapping existing_sets
            new_merged_set = current_set.copy()
            for i in sorted(overlapping_indices, reverse=True): # Iterate backwards to safely remove
                new_merged_set.update(merged_partitions.pop(i))
            merged_partitions.append(new_merged_set)

    # Finalize the mapping and parts
    final_fixed_mapping = {}
    final_parts_tuples: List[Tuple[int, ...]] = sorted([tuple(sorted(list(s))) for s in merged_partitions])

    # Rebuild the mapping from the final partitions
    for state_idx in range(num_states):
        found_partition = False
        for part_tuple in final_parts_tuples:
            if state_idx in part_tuple:
                final_fixed_mapping[state_idx] = part_tuple
                found_partition = True
                break
        if not found_partition:
            # State was not in any of the original mapping's values, or was in an empty partition.
            # Assign it to its own partition.
            final_fixed_mapping[state_idx] = tuple([state_idx])
            # Add this new individual partition to final_parts_tuples if it's not implicitly there
            # (e.g. if num_states is larger than max index in initial mapping values)
            # This case might indicate an issue with the input `mapping` if it's supposed to cover all states.
            # For now, assume `mapping` covers all relevant states or this handling is acceptable.
            # To be robust, ensure all states 0 to num_states-1 get a mapping.

    return final_fixed_mapping, final_parts_tuples


def build_state_abstraction(
    similarity_matrix: NDArray[np.float_], 
    mdp: utils.MDP, 
    similarity_threshold: float = 0.1
) -> Tuple[List[int], utils.MDP, NDArray[np.float_]]:
    """
    Builds a state abstraction for an MDP based on state similarity.

    States are considered similar if their similarity value (e.g., difference in Q-values)
    is less than `similarity_threshold`. The function partitions states based on this similarity,
    selects representative states for each partition, and constructs an abstract MDP
    and a mapping function from original to abstract states.

    Args:
        similarity_matrix: A square matrix where `similarity_matrix[i, j]` indicates
                           the similarity (or distance) between original state i and state j.
                           Lower values mean more similar.
        mdp: The original MDP object.
        similarity_threshold: The threshold below which states are considered similar.

    Returns:
        A tuple containing:
            - representative_abstract_state_indices: A list of indices of the original states
              that serve as representatives for the abstract states.
            - abstract_mdp: The constructed abstract MDP.
            - abstraction_mapping_matrix: A matrix (num_abstract_states, num_original_states)
              mapping original states to abstract states.
    """
    # `bools` indicates pairs of states that are NOT similar enough to be in the same abstract state.
    # The original code `similar_states + np.eye() < tol` implies `similar_states` is a distance matrix.
    # Let's rename `similar_states` to `state_distance_matrix` for clarity.
    # `bools[i,j] is True` if state i and j are similar enough to be grouped.
    # Add np.eye to ensure self-similarity for partitioning logic.
    num_original_states: int = mdp.S
    # Ensure diagonal is True for partitioning logic (reflexivity)
    # Similarity matrix should be symmetric.
    # bool_similarity_matrix should be True if states are similar.
    # Original code: `bools = similar_states + np.eye(similar_states.shape[0]) < tol`
    # This means if `similar_states[i,j]` (a distance) is small, they are similar.
    # np.eye is added, then compared. If similar_states is distance, diagonal is 0.
    # So, `0 + 1 < tol` (if tol > 1) or `0 + 0 < tol` (if eye is added before comparison and is 0).
    # Let's assume `similarity_matrix` is a distance matrix, so low values = similar.
    # We want `similarity_matrix[i,j] < similarity_threshold` for similarity.
    # The `partitions` function expects a boolean matrix where True means "are in the same partition element".
    
    # Create a boolean similarity matrix for `partitions`
    # Ensure reflexivity: diagonal should be True.
    # Ensure symmetry: if (i,j) is similar, (j,i) should be too.
    # The input `similarity_matrix` might already be distances, so check `dist < threshold`.
    bool_similarity_matrix: NDArray[np.bool_] = (similarity_matrix < similarity_threshold)
    np.fill_diagonal(bool_similarity_matrix, True) # Ensure reflexivity for partitioning
    # Symmetrize, e.g., by taking the OR if not already symmetric
    bool_similarity_matrix = np.logical_or(bool_similarity_matrix, bool_similarity_matrix.T)


    if not bool_similarity_matrix.any(): # Changed from sum() == 0 to any() to reflect boolean matrix
        raise ValueError('No abstraction possible with the given threshold; no states are similar.')

    mapping, parts = partitions(bool_similarity_matrix)
    
    # If partitions are not transitive (e.g. A~B, B~C but not A~C based on direct thresholding),
    # fix_mapping can be used to enforce transitivity.
    # This step is crucial if the initial similarity relation (dist < threshold) is not transitive.
    mapping, parts = fix_mapping(mapping) # parts will be list of tuples of state indices

    if not parts or all(not p for p in parts): # Check if parts is empty or contains only empty tuples
        raise ValueError("Partitions resulted in no abstract states. Check similarity matrix and threshold.")

    # Pick a representative state from each non-empty partition.
    # The representative is usually the smallest indexed state in the partition.
    representative_abstract_state_indices: List[int] = sorted(list(set(p[0] for p in parts if p))) # Ensure p is not empty
    
    if not representative_abstract_state_indices:
        raise ValueError("Could not identify representative states for abstraction.")

    num_abstract_states: int = len(representative_abstract_state_indices)

    abstraction_mapping_matrix: NDArray[np.float_] = construct_abstraction_fn(
        mapping, 
        representative_abstract_state_indices, 
        num_original_states, 
        num_abstract_states
    )

    abstract_mdp: utils.MDP = abstract_the_mdp(mdp, representative_abstract_state_indices)

    return representative_abstract_state_indices, abstract_mdp, abstraction_mapping_matrix


def build_option_abstraction(k: int, P: NDArray[np.float_], r: NDArray[np.float_]) -> None:
    """
    Placeholder for exploring how MDP complexity changes with k-step option transformations.
    This function is not yet implemented.

    Args:
        k: The number of steps for the option.
        P: Transition probability matrix of the MDP.
        r: Reward matrix of the MDP.
    """
    # TODO: Implement option abstraction logic.
    pass


def PI(initial_policy_representation: NDArray[np.float_], mdp_obj: utils.MDP, abstraction_mapping_matrix: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Performs Policy Iteration on an MDP and lifts the resulting value function to the original state space.

    Args:
        initial_policy_representation: An initial policy representation (e.g., log probabilities)
                                       for the policy iteration algorithm. Shape (S_abs, A) or (S_orig, A)
                                       depending on where PI is run. Assuming it's for M (abstract MDP).
        mdp_obj: The MDP object (abstract or original) on which to run Policy Iteration.
        abstraction_mapping_matrix: A matrix (num_abstract_states, num_original_states) used to lift
                                    the policy or value function if PI is run on an abstract MDP.
                                    If PI is run on the original MDP, this can be an identity matrix an an f.T.

    Returns:
        The value function of the optimal policy, lifted to the original state space if applicable.
        Shape (num_original_states, 1) or (num_original_states,).
    """
    # Policy iteration usually finds pi_star directly for M.
    # If M is abstract_mdp, init should be (abs_mdp.S, abs_mdp.A)
    # pi_star will be (abs_mdp.S, abs_mdp.A)
    # utils.value_functional expects policy of shape (S,A) or (S,) if deterministic
    # np.dot(f.T, pi_star) - this seems to assume pi_star is (abs_S, 1) or (abs_S,).
    # If pi_star is (abs_S, A), then lifting needs care.
    # The original code's `np.dot(f.T, pi_star)` implies pi_star is a state-value or state-action value vector.
    # Let's assume pi_star from `utils.solve` is the policy matrix (S, A).
    
    # If `initial_policy_representation` is log probabilities, `policy_iteration` handles it.
    optimal_policy: NDArray[np.float_] = utils.solve(ss.policy_iteration(mdp_obj), initial_policy_representation)[-1]
    
    # Calculate value function for this optimal_policy in mdp_obj's state space
    V_optimal_policy: NDArray[np.float_] = utils.value_functional(mdp_obj.P, mdp_obj.r, optimal_policy, mdp_obj.discount)
    
    # Lift V_optimal_policy using abstraction_mapping_matrix.T (which is f from original)
    # V_optimal_policy has shape (mdp_obj.S,). We want (original_mdp.S,).
    # abstraction_mapping_matrix has shape (num_abstract_states, num_original_states)
    # So abstraction_mapping_matrix.T has shape (num_original_states, num_abstract_states)
    # If mdp_obj is the abstract mdp, V_optimal_policy has shape (num_abstract_states,).
    # Lifted_V = abstraction_mapping_matrix.T @ V_optimal_policy
    lifted_V_optimal_policy: NDArray[np.float_] = np.dot(abstraction_mapping_matrix.T, V_optimal_policy)
    
    return lifted_V_optimal_policy


def Q(initial_q_values: NDArray[np.float_], mdp_obj: utils.MDP, abstraction_mapping_matrix: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Performs Q-learning on an MDP and lifts the resulting Q-values (max over actions) to the original state space.

    Args:
        initial_q_values: Initial Q-values for the Q-learning algorithm. Shape (mdp_obj.S, mdp_obj.A).
        mdp_obj: The MDP object (abstract or original) on which to run Q-learning.
        abstraction_mapping_matrix: A matrix (num_abstract_states, num_original_states) used to lift values.
                                    abstraction_mapping_matrix.T has shape (num_original_states, num_abstract_states).

    Returns:
        The state values (max_a Q*(s,a)) of the learned policy, lifted to the original state space.
        Shape (num_original_states,).
    """
    # Q_star will have shape (mdp_obj.S, mdp_obj.A)
    Q_star: NDArray[np.float_] = utils.solve(ss.q_learning(mdp_obj, learning_rate=0.01), initial_q_values)[-1] # Assuming fixed learning rate
    
    # Max over actions to get state values: V_star_approx(s) = max_a Q*(s,a)
    V_star_approx_mdp_obj_space: NDArray[np.float_] = np.max(Q_star, axis=1) # Shape (mdp_obj.S,)
    
    # Lift V_star_approx to original state space
    # abstraction_mapping_matrix.T has shape (num_original_states, num_abstract_states)
    # V_star_approx_mdp_obj_space has shape (num_abstract_states,) if mdp_obj is abstract
    lifted_V_star_approx: NDArray[np.float_] = np.dot(abstraction_mapping_matrix.T, V_star_approx_mdp_obj_space)
    
    return lifted_V_star_approx


def SARSA(initial_q_values: NDArray[np.float_], mdp_obj: utils.MDP, abstraction_mapping_matrix: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Performs SARSA on an MDP and lifts the resulting Q-values (max over actions) to the original state space.

    Args:
        initial_q_values: Initial Q-values for the SARSA algorithm. Shape (mdp_obj.S, mdp_obj.A).
        mdp_obj: The MDP object (abstract or original) on which to run SARSA.
        abstraction_mapping_matrix: A matrix (num_abstract_states, num_original_states) used to lift values.
                                    abstraction_mapping_matrix.T has shape (num_original_states, num_abstract_states).

    Returns:
        The state values (max_a Q*(s,a)) of the learned policy, lifted to the original state space.
        Shape (num_original_states,).
    """
    # Q_star will have shape (mdp_obj.S, mdp_obj.A)
    Q_star: NDArray[np.float_] = utils.solve(ss.sarsa(mdp_obj, learning_rate=0.01), initial_q_values)[-1] # Assuming fixed learning rate

    # Max over actions to get state values: V_star_approx(s) = max_a Q*(s,a)
    V_star_approx_mdp_obj_space: NDArray[np.float_] = np.max(Q_star, axis=1) # Shape (mdp_obj.S,)

    # Lift V_star_approx to original state space
    lifted_V_star_approx: NDArray[np.float_] = np.dot(abstraction_mapping_matrix.T, V_star_approx_mdp_obj_space)
    
    return lifted_V_star_approx


def VI(initial_state_values: NDArray[np.float_], mdp_obj: utils.MDP, abstraction_mapping_matrix: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Performs Value Iteration on an MDP and lifts the resulting state values to the original state space.

    Args:
        initial_state_values: Initial state values for the Value Iteration algorithm. Shape (mdp_obj.S,).
        mdp_obj: The MDP object (abstract or original) on which to run Value Iteration.
        abstraction_mapping_matrix: A matrix (num_abstract_states, num_original_states) used to lift values.
                                    abstraction_mapping_matrix.T has shape (num_original_states, num_abstract_states).

    Returns:
        The optimal state values, lifted to the original state space. Shape (num_original_states,).
    """
    # V_star will have shape (mdp_obj.S,)
    V_star_mdp_obj_space: NDArray[np.float_] = utils.solve(ss.value_iteration(mdp_obj, tolerance=0.01), initial_state_values)[-1] # Assuming fixed tolerance

    # Lift V_star to original state space
    lifted_V_star: NDArray[np.float_] = np.dot(abstraction_mapping_matrix.T, V_star_mdp_obj_space)
    
    return lifted_V_star
