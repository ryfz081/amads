# basics.py - basic symbolic music representation classes
"""
Quick overview: The basic hierarchy of a score is shown here.
Each level of this hierarchy can contain 0
or more instances of the next level. Levels are optional,
allowing for more note-list-like representations:

Score (one per musical work or movement)
    Part (one per instrument)
        Staff (usually 1, but e.g. 2 for grand staff)
            Measure (one for each measure)
                (Measure can contain multiple instances of the following)
                Note
                Rest
                Chord
                    Note (one for each note of the chord)
                KeySignature
                TimeSignature

A "flattened" score looks like this:

Score (one per musical work or movement)
    Part (one per instrument)
        Note (no other instances allowed, no ties)

"""

# TODO: most copy() methods for Mgroup subclasses ignore content.
#    To reduce confusion, rename to copy_empty().
# TODO: deep_copy() calls should use deepcopy() function instead,
#    and deep_copy() methods should be changed to __deepcopy__()
#    to take advantage of existing conventions and copy module.


import functools
from math import floor
import weakref
import sys


from musmart.core.timemap import Timemap, Mapbeat

class Event:
    """Event is a superclass for Note, Rest, Mgroup, and just about
    anything that takes place in time.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any

    def __init__(self, dur, offset):
        self.offset = offset
        self.dur = dur
        self._parent = None

    def copy(self):
        raise Exception(
                "Event is abstract, subclass should override copy()")

    def deep_copy(self):
        raise Exception(
                "Event is abstract, subclass should override deep_copy()")

    @property
    def end_offset(self):
        return self.offset + self.dur

    @property
    def parent(self):
        return self._parent and self._parent

    @parent.setter
    def parent(self, p):
        self._parent = weakref.ref(p)

    def qstart(self):
        """Retrieve the start time in quarters.
        """
        p = self.parent  # save it to prevent race condition
        if p:
            return p().qstart() + self.offset
        else:
            return self.offset

    def qstop(self):
        return self.qstart() + self.dur



class Rest (Event):
    """Rest represents a musical rest. It is normally an element of
    a Measure.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any

    def __init__(self, dur=1, offset=0):
        super().__init__(dur, offset)


    def copy(self):
        """Make a copy of just this object with no parent.
        """
        return Rest(dur=self.dur, offset=self.offset)


    def show(self, indent=0):
        print(' ' * indent, f"Rest at {self.qstart():.3f} ",
              f"offset {self.offset:.3f} dur {self.dur:.3f}", sep='')
        return self


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        r = Rest(self.dur, self.offset)
        return r



class Note (Event):
    """Note represents a musical note. It is normally an element of
    a Measure.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any
    # pitch -- a Pitch
    # dynamic -- None, integer, or string
    # lyric -- None or string
    # tie -- None, 'start', 'stop', or 'continue'. A note that begins a
    #        sequence (or pair) of tied notes uses 'start'. A note that
    #        ends a sequence of tied notes uses 'stop'. A note that is
    #        tied to its predecessor and successor uses 'continue'.
    #        Ties are not slurs. There is no representation for slurs
    #        between notes of different pitches.

    def __init__(self, dur=1, pitch=None, dynamic=None, lyric=None, offset=0):
        """pitch is normally a Pitch, but can be an integer MIDI key number
        that will be converted to a Pitch object.
        """
        super().__init__(dur, offset)
        if isinstance(pitch, Pitch):
            self.pitch = pitch
        elif pitch:
            self.pitch = Pitch(pitch)
        else:
            self.pitch = Pitch.C4
        self.dynamic = dynamic
        self.lyric = lyric
        self.tie = None


    def copy(self):
        """Make a copy of just this object with no parent.
        """
        n = Note(dur=self.dur, pitch=self.pitch, dynamic=self.dynamic,
                 lyric=self.lyric, offset=self.offset)
        n.tie = self.tie
        return n


    def show(self, indent=0):
        tieinfo = ''
        if self.tie != None:
            tieinfo = " tie " + self.tie
        dynamicinfo = ''
        if self.dynamic != None:
            dynamicinfo = " dyn " + str(self.dynamic)
        lyricinfo = ''
        if self.lyric != None:
            lyricinfo = " lyric " + self.lyric
        print(' ' * indent, f"Note at {self.qstart():0.3f} ",
              f"offset {self.offset:0.3f} dur {self.dur:0.3f} pitch ",
              self.name_with_octave, tieinfo, dynamicinfo, lyricinfo, sep='')
        return self


    def deep_copy(self):
        """Make a "deep" copy; pitch is immutable,
        so copyies can share the pitch object with the original
        """
        n = self.copy()
        return n

    @property
    def name(self): return self.pitch.name

    @property
    def name_str(self): return self.pitch.name_str

    @property
    def name_with_octave(self): return self.pitch.name_with_octave

    @property
    def pitch_class(self): return self.pitch.pitch_class

    @pitch_class.setter
    def pitch_class(self, pc): self.pitch.pitch_class = pc

    @property
    def octave(self): return self.octave

    @octave.setter
    def octave(self, oct): self.pitch.octave = oct

    @property
    def keynum(self):
        return self.pitch.keynum

    def enharmonic(self): return self.pitch.enharmonic()

    def upper_enharmonic(self): return self.pitch.upper_enharmonic()

    def lower_enharmonic(self): return self.pitch.lower_enharmonic()



class TimeSignature (Event):
    """TimeSignature is a zero-duration Event with timesig info.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any
    # beat -- the "numerator" of the key signature: beats per measure, a
    #         number, which may be a fraction.
    # beat_type -- the "numerator" of the key signature: a whole number
    #         power of 2, e.g. 1, 2, 4, 8, 16, 32, 64.

    def __init__(self, beat=4, beat_type=4, offset=0):
        super().__init__(0, offset)
        self.beat = beat
        self.beat_type = beat_type


    def copy(self):
        """Make a copy of just this object with no parent.
        """
        return TimeSignature(self.beat, self.beat_type, offset=self.offset)


    def show(self, indent=0):
        print(' ' * indent, f"TimeSignature at {self.qstart():0.3f} offset ",
              f"{self.offset:0.3f}: {self.beat}/{self.beat_type}", sep='')
        return self


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        ts = self.copy()
        return ts



class KeySignature (Event):
    """KeySignature is a zero-duration Event with keysig info.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any
    # keysig -- an integer, number of sharps (if positive) and flats (if
    #         negative), e.g. -3 for Eb major or C minor.

    def __init__(self, keysig=0, offset=0):
        super().__init__(0, offset)
        self.keysig = keysig


    def copy(self):
        """Make a copy, omitting weak link to parent.
        """
        ks = KeySignature(keysig=self.keysig, offset=self.offset)
        return ks


    def show(self, indent=0):
        print(' ' * indent, f"KeySignature at {self.qstart():0.3f} offset ",
              f"{self.offset:0.3f}", abs(self.keysig),
              " sharps" if self.keysig > 0 else " flats", sep='')
        return self


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        return self.copy()


@functools.total_ordering
class Pitch:
    """A Pitch represents a symbolic musical pitch. It has two parts:
    The keynum is a number that corresponds to the MIDI convention
    where C4 is 60, C# is 61, etc. The alt is an alteration, where +1
    represents a sharp and -1 represents a flat. Alterations can also
    be, for example, 2 (double-sharp) or -0.5 (quarter-tone flat).
    The symbolic note name is derived by *subtracting* alt from keynum.
    E.g. C#4 has keynum=61, alt=1, so 61-1 gives us 60, corresponding
    to note name C. A Db has the same keynum=61, but alt=-1, and 61-(-1)
    gives us 62, corresponding to note name D. There is no representation
    for the "natural sign" (other than alt=0, which could imply no
    accidental) or "courtesy accidentals."  Because accidentals normally
    "stick" within a measure or are implied by key signatures, accidentals
    are often omitted in the score presentation. Nonetheless, these
    implied accidentals are encoded in the alt field and keynum is the
    intended pitch with the accidental applied.
    """
    # keynum - MIDI key number, e.g. C4 = 60
    # alt - alteration, e.g. flat = -1

    def __init__(self, kn, alt=0):
        self.keynum = kn
        self.alt = alt
        unaltered = kn - alt
        if int(unaltered) != unaltered:  # not a whole number
            # fix alt so that unaltered is an integer
            diff = unaltered - round(unaltered)
            self.alt -= diff
            unaltered = round(kn - self.alt)
        # make sure pitch class of unaltered is in {C D E F G A B}
        pc = unaltered % 12
        if pc in [6, 1]:  # F#->F, C#->C
            self.alt += 1
        elif pc in [10, 3, 8]:  # Bb->B, Eb->E, Ab->A
            self.alt -= 1
        # now (keynum + alt) % 12 is in {C D E F G A B}

    def astuple(self):
        return (self.keynum, self.alt)

    def __eq__(self, other):
        return self.astuple() == other.astuple()

    def __hash__(self):
        return hash(self.astuple())

    def __lt__(self, other):
        # We sort first by keynum, then by alt.
        # We consider pitches with sharps (i.e. positive alt) to be lower
        # because their letter names are lower in the musical alphabet.
        return (self.keynum, - self.alt) < (other.keynum, - other.alt)

    @property
    def name(self):
        """return one of {0, 2, 4, 5, 7, 9, 11} corresponding to letter names
        """
        return (self.keynum - self.alt) % 12

    @property
    def name_str(self):
        """return string name including accidentals (# or -) if alt is integral
        """
        unaltered = round(self.keynum - self.alt)
        base = ["C", "?", "D", "?", "E", "F", "?", "G", "?", "A", "?", "B"][
                unaltered % 12]
        accidentals = "?"
        if round(self.alt) == self.alt:  # an integer value
            if self.alt > 0:
                accidentals = "#" * self.alt
            elif self.alt < 0:
                accidentals = "b" * -self.alt
            else:
                accidentals = ""  # natural
        return base + accidentals

    @property
    def name_with_octave(self):
        unaltered = round(self.keynum - self.alt)
        octave = (unaltered // 12) - 1
        return self.name_str + str(octave)

    @property
    def pitch_class(self):
        return self.keynum % 12

    @pitch_class.setter
    def pitch_class(self, pc):
        self.keynum = (self.octave + 1) * 12 + pc % 12

    @property
    def octave(self):
        """Returns the octave number ignoring enharmonics. E.g. C4 is enharmonic
        to B#3 and represent the same (more or less) pitch, but BOTH have an
        octave of 4. On the other hand name_str() will return "C4" and "B#3",
        respectively.
        """
        return floor(self.keynum) // 12 - 1

    @octave.setter
    def octave(self, oct):
        self.keynum = (oct + 1) * 12 + self.pitch_class

    def enharmonic(self):
        """"If alt is non-zero, return a Pitch where alt is zero
        or has the opposite sign and where alt is minimized. E.g.
        enharmonic(C-double-flat) is A-sharp (not B-flat). If alt
        is zero, return a Pitch with alt of +1 or -1 if possible.
        Otherwise, return a Pitch with alt of -2.
        """
        alt = self.alt
        unaltered = round(self.keynum - alt)
        if alt < 0:
            while alt < 0 or (unaltered % 12) not in [0, 2, 4, 5, 7, 9, 11]:
                unaltered -= 1
                alt += 1
        elif alt > 0:
            while alt > 0 or (unaltered % 12) not in [0, 2, 4, 5, 7, 9, 11]:
                unaltered += 1
                alt -= 1
        else:  # alt == 0
            unaltered = unaltered % 12
            if unaltered in [0, 5]:  # C->B#, F->E#
                alt = 1
            elif unaltered in [11, 4]:  # B->Cb, E->Fb
                alt = -1
            else:  # A->Bbb, D->Ebb, G->Abb
                alt = -2
        return Pitch(self.keynum, alt=alt)

    def upper_enharmonic(self):
        """"return a valid Pitch with alt decreased by 1 or 2, e.g. C#->Db,
        C##->D, C###->D#.
        """
        alt = self.alt
        unaltered = round(self.keynum - alt) % 12
        if unaltered in [0, 2, 5, 7, 9]:  # C->D, D->E, F->G, G->A, A->B
            alt -= 2
        else:  # E->F, B->C
            alt -= 1
        return Pitch(self.keynum, alt=alt)

    def lower_enharmonic(self):
        """"return a valid Pitch with alt increased by 1 or 2, e.g. Db->C#,
        D->C##, D#->C###
        """
        alt = self.alt
        unaltered = round(self.keynum - alt) % 12
        if unaltered in [2, 4, 7, 9, 11]:  # D->C, E->D, G->F, A->G, B->A
            alt += 2
        else:  # F->E, C->B
            alt += 1
        return Pitch(self.keynum, alt=alt)



class Mgroup (Event):
    """An Mgroup is a collection of Event objects. This is an abstract
    class. Use one of the subclasses: Sequence or Concurrence.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any
    # content -- elements contained within this collection

    def __init__(self, offset, dur, content):
        super().__init__(dur, offset)
        self.content = [] if content == None else content


    def copy(self):
        raise Exception(
                "Mgroup is abstract, subclass should override copy()")


    def show(self, indent=0, label="Mgroup"):
        print(' ' * indent, label, f" at {self.qstart():0.3f} offset ",
              f"{self.offset:0.3f} dur {self.dur:0.3f}", sep='')
        for elem in self.content:
            elem.show(indent + 4)
        return self


    @property
    def last(self):
        return self.content[-1] if len(self.content) > 0 else None


    def deep_copy(self):
        raise Exception(
                "Mgroup is abstract, subclass should override deep_copy()")


    def has_chords(self):
        """Test if Mgroup (e.g. Score, Part, Staff, Measure) has any
        Chord objects.
        """
        chords = self.find_all(Chord)
        # if there are no chords, next will return "empty"
        return next(chords, "empty") != "empty"


    def has_ties(self):
        """Test if Mgroup (e.g. Score, Part, Staff, Measure) has any
        tied notes.
        """
        notes = self.find_all(Note)
        for note in notes:
            if note.tie:
                return True
        return False


    def has_measures(self):
        """Test if Mgroup (e.g. Score, Part, Staff) has any measures."""
        measures = self.find_all(Measure)
        # if there are no chords, next will return "empty"
        return next(measures, "empty") != "empty"


    def insert(self, event):
        """insert an event without any changes to event.offset or
        self.dur. If the event is out of order, insert it just before
        the first element with a greater offset. This method is similar
        to append(), but it has different defaults for offset and
        update_dur.
        """
        if self.last and event.offset < self.last.offset:
            # search in reverse from end
            i = len(self.content) - 2
            while i >= 0 and self.content[i].offset > event.offset:
                i -= 1
            # now i is either -1 or content[i] <= event.offset, so
            # insert event at content[i+1]
            self.content.insert(i + 1, event)
        else:  # simply append at the end of content:
            self.content.append(event)
        event.parent = self
        return self


    def find_all(self, elemType):
        """This is a generator that returns all contained instances of
        type using depth-first search.
        """
        for elem in self.content:
            if isinstance(elem, elemType):
                yield elem
            elif isinstance(elem, Mgroup):
                yield from elem.find_all(elemType)



class Sequence (Mgroup):
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any
    # content -- elements contained within this collection

    def __init__(self, offset=0, dur=None, content=None):
        """Sequence represents a temporal sequence of music events.
        dur(ation) defaults to the duration of provided content or 0
        if content is empty or None.
        """
        if content == None:
            content = []
        if dur == None:
            if len(content) == 0:
                dur = 0
            else:
                dur = content[-1].end_offset
        super().__init__(offset, dur, content)


    def copy(self):
        """Make a copy, omitting weak link to parent.
        """
        s = Sequence(offset=self.offset, dur=self.dur)
        return s


    def show(self, indent=0, label="Sequence"):
        return super().show(indent, label)


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        s = self.copy()
        for event in self.content:
            s.insert(event.deep_copy())
        return s


    def last_qstop(self):
        """return the end time (in quarters) of the last element,
        or the Sequence start time if the Sequence is empty
        """
        if len(self.content) == 0:
            return self.qstart()
        else:
            return self.last.last_qstop()


    def last_end_offset(self):
        """return the end_offset (in quarters) of the last element,
        or 0 if the Sequence is empty
        """
        if len(self.content) == 0:
            return 0
        else:
            return self.last.last_end_offset()


    def append(self, element, offset=None, update_dur = True):
        """Append an element. If offset is specified, the element is
        modified to start at this offset, and the duration of self
        is unchanged. If offset is not specified or None, the element
        offset is changed to the dur(ation) of self, which is then
        incremented by the duration of element.
        """
        if offset == None:
            element.offset = self.dur
            if update_dur:
                self.dur += element.dur
        else:
            element.offset = offset
        self.insert(element)  # places element in order and sets element parent


    def pack(self):
        """Adjust the content to be sequential, with zero offset for the
        first element, and each other object at an offset equal to the
        end_offset of the previous element. The dur(ation) of self is set
        to the end_offset of the last element. This method essentially,
        arranges the content to eliminate gaps. pack() works recursively
        on elements that are Mgroups.
        """
        offset = 0
        for elem in self.content:
            elem.offset = 0
            if isinstance(elem, Mgroup):
                elem.pack()
            offset += elem.dur



class Concurrence (Mgroup):
    """Concurrence represents a temporally simultaneous collection
    of music events (but if elements have a non-zero offset, a Concurrence
    can represent events organized over time).  Thus, the main distinction
    between Concurrence and Sequence is the behavior of methods, since both
    classes can represent simultaneous or sequential events.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # content -- elements contained within this collection

    def __init__(self, offset=0, dur=None, content=None):
        """dur(ation) defaults to the maximum end_offset of provided content
        or 0 if content is empty.
        """
        if content == None:
            content = []
        if dur == None:
            dur = 0
            for elem in content:
                dur = max(dur, elem.end_offset)
        super().__init__(offset, dur, content)


    def copy(self):
        """Make a copy, omitting weak link to parent.
        """
        c = Concurrence(offset=self.offset, dur=self.dur)
        return c


    def show(self, indent=0, label='Concurrence'):
        return super().show(indent, label)


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        c = self.copy()
        for event in self.content:
            c.insert(event.deep_copy())
        return c


    def pack(self):
        """Adjust the content to offsets of zero. The dur(ation) of self
        is set to the maximum end_offset of the elements. This method
        essentially, arranges the content to eliminate gaps. pack() works
        recursively on elements that are Mgroups.
        """
        self.dur = 0
        for elem in self.content:
            elem.offset = 0
            if isinstance(elem, Mgroup):
                elem.pack()
            self.dur = max(self.dur, elem.dur)


    def append(self, element, offset=0, update_dur=True):
        """Append an element to the content with the given offset.
        (Specify offset=element.offset to retain the element's offset.)
        By default, the dur(ation) of self is increased to the
        end_offset of element if the end_offset is greater than the
        current dur(ation). To retain the dur(ation) of self, specify
        update_dur=False.
        """
        element.offset = offset
        self.insert(element)
        if update_dur:
            self.dur = max(self.dur, element.end_offset)



class Chord (Concurrence):
    """A Chord is a collection of Notes, normally with offsets of zero
    and the same durations and distinct keynums, but this is not enforced.
    The order of notes is arbitrary. Normally, a Chord is a member of a
    Staff. There is no requirement that simultaneous or overlapping notes
    be grouped into Chords, so the Chord class is merely an optional
    element of music structure representation. Representation note: An
    alternative representation would be to subclass Note and allow a
    list of pitches, which has the advantage of enforcing the shared
    offsets and durations. However, there can be ties connected
    differently to each note within the Chord, thus we use a Concurrence
    with Note objects as elements. Each Note.tie can be different.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # _parent -- weak reference to containing object if any
    # content -- elements contained within this collection

    def show(self, indent=0):
         return super().show(indent, "Chord")


    def copy(self):
        return Chord(offset=self.offset, dur=self.dur)


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        m = self.Copy()
        for event in self.content:
            m.insert(event.deep_copy())
        return m



class Measure (Sequence):
    """A Measure models a musical measure (bar) and can contain many object
    types including Note, Rest, Chord, KeySignature, Timesignature. Measures
    are elements of a Staff.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # content -- elements contained within this collection
    # number -- A string or None. The number assigned to the measure in the
    #         score (if any). E.g. "22a".

    def __init__(self, number=None, offset=0, dur=4, content=None):
        super().__init__(offset, dur, content)
        self.number = number


    def copy(self):
        """Make a copy, omitting weak link to parent.
        """
        m = Measure(number=self.number, offset=self.offset, dur=self.dur)
        return m


    def show(self, indent=0):
        nstr = ' ' + str(self.number) if self.number else ''
        return super().show(indent, "Measure" + nstr)


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        m = self.copy()
        for event in self.content:
            m.insert(event.deep_copy())
        return m


    def is_measured(self):
        """Test if Measure is well-formed. Conforms to strict hierarchy of:
        Measure-(Note or Chord-Note)
        """
        for measure in self.content:
            # Measure can contain many object types, so instead of
            # testing for legal content, we look for things that are
            # illegal content:
            if isinstance(measure, Staff) or isinstance(measure, Part) or \
               isinstance(measure, Score):
                return False
        return True


    def strip_chords(self):
        """return a deep copy with Chord elements replaced by individual notes.
        """
        measure = self.copy()
        for elem in self.content:
            if isinstance(elem, Chord):
                for note in elem:
                    measure.insert(note.deep_copy())
            else:
                measure.insert(elem.deep_copy())
        return measure



def note_qstart(note):
    """helper function to sort notes
    """
    return note.qstart()



class Score (Concurrence):
    """A score is a top-level object representing a musical work.
    Normally, a Score contains Part objects, all with offset zero.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # content -- elements contained within this collection
    # timemap -- a map from quarters to seconds (or seconds to quarters)
    #
    # Additional attributes may be assigned, e.g. 'title', 'source_file',
    # 'composer', etc. (TBD)

    def __init__(self, offset=0, dur=0, content=None, timemap=None):
        super().__init__(offset, dur, content)
        self.timemap = timemap if timemap else Timemap()


    def copy(self):
        """Make a copy, omitting weak link to parent.
        """
        s = Score(offset=self.offset, dur=self.dur,
                  timemap=self.timemap)
        return s


    def show(self, indent=0):
        print(' ' * indent, f"Score at {self.qstart():0.3f} offset ",
              f"{self.offset:0.3f} dur {self.dur:0.3f}", sep='')
        self.timemap.show(indent + 4)
        for elem in self.content:
            elem.show(indent + 4)
        return self


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        s = self.copy()
        s.timemap = self.timemap.deep_copy()
        for event in self.content:
            # deep copy each component into s
            s.insert(event.deep_copy())
        return s


    def is_measured(self):
        """Test if Score is measured. Conforms to strict hierarchy of:
        Score-Part-Staff-Measure-(Note or Chord-Note)
        """
        for part in self.content:
            if not isinstance(part, Part):
                return False
            if not part.is_measured():
                return False
        return True



    def strip_ties(self):
        """Create a new Score with tied note sequences replaced by
        equivalent notes in each staff
        """
        score = self.copy()
        for part in self.content:
            score.insert(part.strip_ties())
        return score


    def strip_chords(self):
        """Create a new Score with chords replaced by
        individual notes in each staff
        """
        score = self.copy()
        for part in self.content:
            score.insert(part.strip_chords())
        return score

    def strip_measures(self):
        # TODO
        pass

    def note_containers(self):
        """Returns a list of note containers. For Measured Scores, these
        are the Staff objects. For Flattened Scores, these are the Part
        objects. This is mainly useful for extracting note sequences where
        each staff represents a separate sequence. In a Flattened Score,
        staves are collapsed and each Part (instrument) represents a
        separate sequence.
        """
        containers = []
        for part in self.content:
            if len(part.content) > 0 and isinstance(part.content[0], Staff):
                containers += part.content
            else:
                containers.append(part)
        return containers

    def is_flattened(self):
        # TODO
        pass

    def is_flattened_and_collapsed(self):
        """Determine if score has been flattened into one part"""
        return self.part_count() == 1 and \
               (len(self.content[0].content) == 0 or  # no notes
                isinstance(self.content[0].content[0], Note))  # Part has notes

    def part_count(self):
        """How many parts are in this score?"""
        return len(self.content)


    def flatten(self, collapse=False):
        """Deep copy notes in a score to a flattened score consisting of
        only Parts containing Notes. If collapse is True, multiple parts are
        collapse into a single part, and notes are ordered according to
        onset times
        """
        score = self.copy()
        score_no_ties = self.strip_ties() # strip ties
        if collapse:  # similar to Part.flatten() but we have to sort and
                      # do some other extra work to put all notes into score
            score_start = score.qstart()
            new_part = Part()
            max_end_offset = 0
            for part in score_no_ties.content:
                for note in part.find_all(Note):
                    note_copy = note.deep_copy()
                    # note offset is now relative to start of part:
                    note_copy.offset = note.qstart() - score_start
                    max_end_offset = max(max_end_offset, note_copy.end_offset)
                    new_part.insert(note_copy)
            new_part.content = sorted(new_part.content, key=note_qstart)
            score.insert(new_part)
            score.dur = score.qstart() + max_end_offset
        else:
            for part in self.content:
                score.insert(part.flatten())
        return score


    def collapse_parts(self, part=None, staff=None):
        """return a flattened score with all parts merged into one and
        sorted in order of note onset times. The returned score has the
        form:
            Score
                Part
                    Note Note Note ...
        so the note list can be accessed using
            score.collapse_parts(...).content[0].content

        If part is given, only notes from the selected part are included.
            part may be an integer to match a part number
            part may be a string to match a part instrument
            part may be a list with an index, e.g. [3] will select
                the 4th part (with zero-based indexing)
        If staff is given, only the notes from selected staves are included.
            staff may be an integer to match a staff number
            staff may be a list with an index, e.g. [1] will select
                the 2nd staff.
        If staff is given without a part specification, an exception
            is raised.
        If staff is given and this is a flattened score (no staves),
            an exception is raised.
        Note: The use of the form [1] for part and staff index notation
            is not ideal, but we need to distinguish between part numbers
            (arbitrary labels) and part index. Initially, I used tuples,
            but they are error prone. E.g. part=(0) means part=0, so you
            have to write keynum_list(part=((0))). With [n], you write
            keynum_list(part=[0]) to indicate an index. This is
            prettier and less prone to error.
        """
        keynums = []
        mtn = self.strip_ties()  # makes a copy we can manipulate
        parts = mtn.content
        mtn.content = []  # reconstruct with only selected parts
        for i, p in enumerate(parts):
            if part is None or (isinstance(part, int) and part == p.number) or \
               (isinstance(part, str) and part == p.instrument) or \
                isinstance(part, list) and part[0] == i:
                if staff is not None:  # select staves
                    if len(p.content[0]) > 0 and \
                       not isinstance(p.content[0], Staff):
                        raise Exception("Expected Part to contain Staff")
                    else:
                        # select staves
                        staves = p.content
                        p.content = []
                        for i, s in enumerate(staves):
                            if staff is None or \
                               (isinstance(staff, int) and \
                                       staff == s.number) or \
                                isinstance(staff, list) and staff[0] == i:
                                p.insert(s)
                if len(p.content) > 0:  # keep part only if there is content
                    mtn.insert(p)
        # Flatten to get selected notes in order of onset time
        return mtn.flatten(collapse=True)




class Part (Concurrence):
    """A Part models a staff or staff group such as a grand staff. For that
    reason, a Part contains one or more Staff objects. It should not contain
    any other object types. Parts are normally elements of a Score.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # content -- elements contained within this collection
    # number -- None or an integer. Normally, the Parts are numbered according
    #         to their top-to-bottom ordering in the Score, starting with 1.
    # instrument -- None or a string naming the instrument to play this Part.

    def __init__(self, number=None, instrument=None, offset=0, dur=0,
                 content=None):
        super().__init__(offset, dur, content)
        self.number = number
        self.instrument = instrument


    def copy(self):
        """Make a copy, omitting weak link to parent.
        """
        p = Part(number=self.number, instrument=self.instrument,
                 offset=self.offset, dur=self.dur)
        return p


    def show(self, indent=0):
        nstr = (' ' + str(self.number)) if self.number else ''
        name = (' (' + self.instrument + ')') if self.instrument else ''
        return super().show(indent, "Part" + nstr + name)


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        p = self.copy()
        for event in self.content:
            # deep copy each component into p
            p.insert(event.deep_copy())
        return p


    def is_measured(self):
        """Test if Part is measured. Conforms to strict hierarchy of:
        Part-Staff-Measure-(Note or Chord-Note)
        """
        for staff in self.content:
            if not isinstance(staff, Staff):
                return False
            if not staff.is_measured():
                return False
        return True


    def strip_ties(self):
        """Create a new Part with tied note sequences replaced by
        equivalent notes in each staff
        """
        part = self.copy()

        for staff in self.content:
            if not isinstance(staff, Staff):
                return self  # no need to strip ties from flattened score
            part.insert(staff.strip_ties())
        return part


    def strip_chords(self):
        """return a deep copy with Chord elements replaced by individual notes.
        """
        part = self.copy()
        for staff in self.content:
            part.insert(staff.strip_chords())
        return part


    def flatten(self):
        """Deep copy notes in a part to a flattened part consisting of
        only Notes.
        """
        part = self.strip_ties()
        flat = self.copy()
        part_start = self.qstart()
        for note in part.find_all(Note):
            note_copy = note.deep_copy()
            # note offset is now relative to start of part:
            note_copy.offset = note.qstart() - part_start
            flat.insert(note_copy)
        return flat



class Staff (Sequence):
    """A Staff models a musical staff line extending through all systems.
    It can also model one channel of a standard MIDI file track. A Staff
    normally contains Measure objects and is an element of a Part.
    """
    # offset -- start time in quarters as an offset from parent's start time
    # dur -- duration in quarters
    # content -- elements contained within this collection
    # number -- Normally a Staff is given an integer number where 1 is the
    #         top staff of the part, 2 is the 2nd, etc. number may be None.

    def __init__(self, number=None, offset=0, dur=0, content=None):
        super().__init__(offset, dur, content)
        self.number = number


    def copy(self):
        """Make a shallow copy of just this object with no parent.
        """
        return Staff(number=self.number, offset=self.offset, dur=self.dur)


    def show(self, indent=0):
        nstr = (' ' + str(self.number)) if self.number else ''
        return super().show(indent, "Staff" + nstr)


    def deep_copy(self):
        """Make a deep copy, omitting weak link to parent.
        """
        s = self.copy()
        for event in self.content:
            s.insert(event.deep_copy())
        return s


    def is_measured(self):
        """Test if Staff is measured. Conforms to strict hierarchy of:
        Staff-Measure-(Note or Chord-Note)
        """
        for measure in self.content:
            # Staff can contain many objects such as key signature or
            # time signature, so instead of testing for legal content,
            # we look for things that are illegal content:
            if isinstance(measure, Note) or isinstance(measure, Chord) or \
               isinstance(measure, Staff) or isinstance(measure, Part) or \
               isinstance(measure, Score):
                return False
            if not measure.is_measured():
                return False
        return True


    def strip_ties(self):
        """Create a new staff with tied note sequences replaced by
        equivalent notes
        """
        staff = self.copy()
        for m_num, m in enumerate(self.content):
            measure = m.copy()
            for event in m.content:
                if isinstance(event, Note):
                    if event.tie is None:
                        measure.insert(event.copy())
                    elif event.tie == 'start':
                        new_event = event.copy()
                        new_event.dur = self.tied_dur(event, m_index=m_num)
                        new_event.tie = None
                        measure.insert(new_event)
                elif isinstance(event, Chord):
                    new_chord = event.copy()
                    for note in event.content:
                        if note.tie is None:
                            new_chord.insert(note.copy())
                        elif note.tie == 'start':
                            new_note = note.copy()
                            new_note.dur = self.tied_dur(note, m_index=m_num)
                            new_note.tie = None
                            new_chord.insert(new_note)
                    measure.insert(new_chord)
                else:    # non-note objects are simply copied:
                    measure.insert(event.deep_copy())
            staff.insert(measure)
        return staff


    def tied_dur(self, note, m_index=None):
        """Compute the full duration of note as the sum of notes that note
        is tied to. note.tie must be 'start'
        """
        measure = note.parent()
        # if note was in chord we need the note's grandparent:
        if isinstance(measure, Chord):
            measure = measure.parent()
        if m_index == None:  # get measure index
            m_index = self.content.index(measure)
        n_index = measure.content.index(note) + 1 # get note index
        qstart = note.qstart()
        # search across all measures for tied-to note:
        while m_index < len(self.content):  # search all measures
            measure = self.content[m_index]
            # search within measure for tied-to note:
            while n_index < len(measure.content):
                event = measure.content[n_index];
                if isinstance(event, Note) and \
                   event.keynum == note.keynum:
                    if event.tie == 'stop':
                            return event.qstop() - qstart
                    elif event.tie != 'continue':
                        raise Exception("inconsistent tie attributes or notes")
                elif isinstance(event, Chord):
                    # search within chord for tied-to note:
                    for n in event.content:
                        if n.keynum == note.keynum:
                            # add durations until 'stop'
                            if n.tie == 'continue' or n.tie == 'stop':
                                dur = n.end_event - note.offset
                                if n.tie == 'stop':
                                    return dur
                n_index += 1
            n_index = 0
            m_index += 1
        raise Exception("incomplete tie")


    def strip_chords(self):
        """return a deep copy with Chord elements replaced by individual notes.
        """
        staff = self.copy()
        for measure in self.content:
            staff.insert(measure.strip_chords())
        return staff
