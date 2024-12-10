import matplotlib.pyplot as plt

from amads.music import example
from amads.pianoroll import pianoroll
from amads.pt_midi_import import partitura_midi_import
from amads.skyline import skyline

my_midi_file = example.fullpath("midi/chopin_prelude_7.mid")


print("------- input from partitura")
myscore = partitura_midi_import(my_midi_file, ptprint=False)
print("------- finished input from partitura")
myscore.show()

pianoroll(myscore)

print("------- Find skyline")
sl = skyline(myscore)

# print(sl)
sl.show()

pianoroll(sl)
plt.show()
