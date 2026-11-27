
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