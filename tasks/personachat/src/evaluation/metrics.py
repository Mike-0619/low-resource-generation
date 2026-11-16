# src/evaluation/metrics.py

def calculate_ranking_metrics(losses, gold_index):
    """
    Ranks the candidates based on their loss and calculates the rank of the gold answer.
    Lowest loss gets rank 1.
    """
    # Sort indices based on loss values (ascending)
    sorted_indices = sorted(range(len(losses)), key=lambda idx: losses[idx])
    
    # Find where the correct (gold) answer placed in the sorted list
    gold_rank = sorted_indices.index(gold_index) + 1
    pred_index = sorted_indices[0]
    
    return gold_rank, pred_index

def aggregate_scores(results_detail, total):
    """
    Computes the final Hits@1, Hits@5, and MRR percentages/scores.
    """
    hits_at_1 = sum(1 for res in results_detail if res["gold_rank"] == 1)
    hits_at_5 = sum(1 for res in results_detail if res["gold_rank"] <= 5)
    mrr_sum = sum(1.0 / res["gold_rank"] for res in results_detail)

    return {
        "hits_at_1_pct": (hits_at_1 / total) * 100,
        "hits_at_5_pct": (hits_at_5 / total) * 100,
        "mrr_score": mrr_sum / total,
        "hits_at_1_count": hits_at_1,
        "hits_at_5_count": hits_at_5
    }