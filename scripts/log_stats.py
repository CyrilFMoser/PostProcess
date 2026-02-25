import re
from collections import defaultdict

log_file = "/users/cymoser/projects/PostProcess/logs/preprocess_timing.log"

# Store totals and counts
totals = defaultdict(float)
counts = defaultdict(int)

# Regex to capture "Label: number s"
pattern = re.compile(r"([A-Za-z ]+):\s*([0-9.]+)s")

with open(log_file, "r") as f:
    for line in f:
        matches = pattern.findall(line)
        for label, value in matches:
            label = label.strip()  # clean up spaces
            value = float(value)
            totals[label] += value
            counts[label] += 1

print("Averages:\n")
for label in sorted(totals.keys()):
    avg = totals[label] / counts[label]
    print(f"{label:10s} | avg = {avg:.6f}s over {counts[label]} steps")
