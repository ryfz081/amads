import matplotlib.pyplot as plt

from amads.music import example
from amads.pianoroll import pianoroll
from amads.pt_midi_import import partitura_midi_import
from amads.segmentgestalt import segmentgestalt

my_midi_file = example.fullpath("midi/sarabande.mid")


print("------- input from partitura")
myscore = partitura_midi_import(my_midi_file, ptprint=False)
print("------- finished input from partitura")

fig = pianoroll(myscore)

print("------- Executing segmentgestalt")
clang_offsets, segment_offsets = segmentgestalt(myscore)
print(clang_offsets)
print(segment_offsets)
xmin, xmax, ymin, ymax = plt.axis()

plt.vlines(clang_offsets, ymin, ymax, colors="purple", label="clang boundary offsets")
plt.vlines(
    segment_offsets, ymin, ymax, colors="green", label="segment boundary offsets"
)
plt.legend(loc="best")

plt.show()
