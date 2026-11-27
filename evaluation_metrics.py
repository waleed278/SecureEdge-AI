
# Results from our first baseline experiment

labels = ["NETWORK", "ACCOUNT", "HARDWARE", "SOFTWARE"]

actual = [
    "NETWORK", "NETWORK",
    "ACCOUNT", "ACCOUNT",
    "HARDWARE", "HARDWARE",
    "SOFTWARE", "SOFTWARE"
]

predicted = [
    "ACCOUNT", "ACCOUNT",
    "ACCOUNT", "ACCOUNT",
    "ACCOUNT", "ACCOUNT",
    "ACCOUNT", "SOFTWARE"
]

# Step 1: Create empty confusion matrix
matrix = {
    label: {other: 0 for other in labels}
    for label in labels
}

# Step 2: Count actual vs predicted combinations
for true_label, predicted_label in zip(actual, predicted):
    matrix[true_label][predicted_label] += 1

# Step 3: Display confusion matrix
print("\nCONFUSION MATRIX")
print(f"{'Actual':<12}" + "".join(f"{x[:4]:>10}" for x in labels))

for label in labels:
    print(
        f"{label:<12}" +
        "".join(f"{matrix[label][p]:>10}" for p in labels)
    )

# Step 4: Calculate accuracy
correct = sum(a == p for a, p in zip(actual, predicted))
accuracy = correct / len(actual)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Step 5: Calculate metrics for every category
f1_scores = []

for label in labels:

    tp = matrix[label][label]

    fp = sum(
        matrix[other][label]
        for other in labels if other != label
    )

    fn = sum(
        matrix[label][other]
        for other in labels if other != label
    )

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    f1_scores.append(f1)

    print(
        f"{label}: Precision={precision:.2f}, "
        f"Recall={recall:.2f}, F1={f1:.2f}"
    )

macro_f1 = sum(f1_scores) / len(f1_scores)
print(f"\nMacro-F1: {macro_f1 * 100:.2f}%")
