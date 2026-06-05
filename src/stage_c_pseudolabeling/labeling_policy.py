from typing import List
import numpy as np

class LabelingPolicy:
    """Base class for labeling policies.

    A labeling policy defines how to convert model scores into final labels
    for pseudo-labeling.

    Subclasses should implement the `decide_labels` method.
    """

    def __init__(self, policy: str = "hardlabeling"):
        self.policy = policy
        
        

    def decision_from_score(self, score: float, threshold: float) -> int:
        return int(float(score) >= float(threshold))


    def pseudo_vector_from_decision(self, decision: int, active_vector: List[int]) -> List[int]:
        """Map a global decision onto an attribute vector:
        - For indices where active_vector[i] == 1 -> set to decision (0/1)
        - Else -> set to -1 (unknown)
        """
        
        v = np.asarray(active_vector).astype(int)
        out = np.full_like(v, -1)
        out[v == 1] = int(decision)
        return out.tolist()

    def decide_labels(self, scores: List[float], threshold: float, active_vectors: List[List[int]]) -> List[List[int]]:
        """Decide final pseudo-labels based on scores, thresholds, and active attribute vectors.

        Parameters:
        - scores: List of model scores for each sample.
        - thresholds: List of thresholds for each attribute.
        - active_vectors: List of active attribute vectors for each sample.

        Returns:
        - List of pseudo-label vectors for each sample.
        """
        labels=[]
        decisions=[]
        if self.policy == "hardlabelling":
            
            for score, active_vector in zip(scores, active_vectors):
                decision_from_score = self.decision_from_score(score, threshold)
                pseudo_vector_from_decision = self.pseudo_vector_from_decision(decision_from_score, active_vector)
                labels.append(pseudo_vector_from_decision)
                decisions.append(decision_from_score)
            return labels, decisions
        else:
            raise NotImplementedError(f"Labeling policy '{self.policy}' is not implemented.")
