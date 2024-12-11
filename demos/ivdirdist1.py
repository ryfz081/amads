"""
Test the `ivdirdist1` function
"""

import matplotlib.pyplot as plt

from amads.ivdirdist1 import ivdirdist1
from amads.music import example
from amads.pt_midi_import import partitura_midi_import

my_midi_file = example.fullpath("/midi/sarabande.mid")

print("------- input from partitura")
myscore = partitura_midi_import(my_midi_file, ptprint=False)
print("------- finished input from partitura")
myscore.show()

print("------- Calculate pitch-class distribution")
id = ivdirdist1(myscore, weighted=True)

print(id)

# Plot the interval distribution
interval_names = [
    "m2",
    "M2",
    "m3",
    "M3",
    "P4",
    "d5",
    "P5",
    "m6",
    "M6",
    "m7",
    "M7",
    "P8",
]
plt.bar(
    interval_names,
    height=[abs(i - 0.5) if i != 0 else 0 for i in id],
    bottom=[min(0.5, i) if i != 0 else 0.5 for i in id],
    color="skyblue",
)
plt.ylim(0, 1)
plt.xlabel("Interval")
plt.ylabel("Probability")
plt.title("Interval Distribution")

plt.show()
