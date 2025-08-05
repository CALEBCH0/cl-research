def get_accuracies(eval_results):
    """Extract accuracies from evaluation results."""
    accuracies = []
    for key, value in sorted(eval_results.items()):
        if 'Top1_Acc_Exp' in key and 'eval_phase' in key:
            accuracies.append(value)
    return accuracies

def get_average_accuracy(eval_results):
    """Calculate the average accuracy from evaluation results."""
    accuracies = get_accuracies(eval_results)
    return sum(accuracies) / len(accuracies) if accuracies else 0.0

def print_results(eval_results, name=""):
    """Print evaluation results in a formatted way."""
    accuracies = get_accuracies(eval_results)
    if name:
        print(f"\n{name}:")
    for i, acc in enumerate(accuracies):
        print(f"Experience {i + 1}: {acc:.4f}")
    if accuracies:
        print(f"Average: {get_average_accuracy(eval_results):.4f}")
        