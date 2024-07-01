import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import simpleaudio as sa
import sounddevice as sd
import soundfile as sf
from basic_pitch.inference import predict
from mido import MetaMessage, MidiFile, bpm2tempo
from music21 import chord, converter, harmony, note, pitch, stream
from pedalboard import Compressor, Gain, Limiter, Pedalboard
from pydub import AudioSegment
from scipy.io.wavfile import write

from qa_util import generate_chord_progression, generate_excerpt


### Main function
def record_and_grade(in_time=True, fs=44100, excerpt=None, exact=True, bpm=120, count_in=2, score_threshold=0.8):

    quarter_bpm = calculate_quarter_bpm(bpm, excerpt.flat.timeSignature) # Calculate the quarter note BPM for the MIDI file
    full_duration = get_score_duration(excerpt, bpm, count_in=count_in)  # seconds
    highest_freq, lowest_freq = get_score_pitch_range(excerpt) # Hz
    min_note_length_ms = get_score_min_note_length(excerpt, bpm) # milliseconds
    board = Pedalboard([Compressor(threshold_db=-20, ratio=8, attack_ms=10, release_ms=min_note_length_ms), 
                        Gain(gain_db=8), Limiter(threshold_db=-10, release_ms=min_note_length_ms)])
    written_chords = True if len([e for e in excerpt.flatten().getElementsByClass(chord.Chord).stream() if 'ChordSymbol' not in e.classSet]) > 0 else False

    # Record user audio and save as WAV file
    user_recording, metronome_np_array = record_with_metronome(excerpt, bpm, count_in, fs, full_duration) if in_time else record_without_metronome(fs=fs)
    output_filename_wav = 'user_recordings/output.wav'
    write(output_filename_wav, fs, user_recording)

    # Process audio
    processed_wav_file = process_audio(output_filename_wav, board, metronome_np_array)

    # Extract base file name and create MIDI file name
    base_filename = os.path.splitext(processed_wav_file)[0]
    output_filename_midi = f'{base_filename}.mid'

    # Basic Pitch
    model_output, midi_data, note_events = predict(processed_wav_file,
                                                   onset_threshold=0.1 if in_time else 0.7,
                                                   frame_threshold=0.2 if in_time else 0.3 if written_chords else 0.5, 
                                                   midi_tempo=quarter_bpm if in_time else 120,
                                                   maximum_frequency=highest_freq,
                                                   minimum_frequency=lowest_freq,
                                                   minimum_note_length=min_note_length_ms if in_time else 50,
                                                   melodia_trick=True,
                                                   debug_file='debug.txt'
                                                   )

    # Write and format MIDI file
    midi_data.write(output_filename_midi)
    user_midi = format_user_midi(MidiFile(output_filename_midi), 
                                 excerpt, 
                                 bpm if in_time else 120,
                                 output_filename_midi
                                 )

    # Convert MIDI file to music21 stream and format
    user_m21_stream = converter.parse(output_filename_midi)
    user_m21_stream_formatted = format_user_stream(user_m21_stream, excerpt, count_in, in_time)
    user_m21_stream_formatted.show()

    # Compare music21 streams - True if the user played the excerpt correctly, False if not
    return compare_m21_streams(excerpt, user_m21_stream_formatted, exact, in_time, score_threshold)

### Music21 functions
def compare_m21_streams(original_stream, user_stream, exact=True, in_time=False, score_threshold=0.8):
    if user_stream is None or len(user_stream.flatten().getElementsByClass(note.Note).stream()) == 0:
        raise ValueError('User stream is empty')
    
    original_notes_list = [o for o in original_stream.flatten().getElementsByClass([note.Note, chord.Chord]).stream() if 'ChordSymbol' not in o.classSet]

    if in_time:
        if original_stream.timeSignature is None:
            raise ValueError('Original stream does not have a time signature')
        
        ### Iterate over each beat in the original stream and determine a percentage-based score based on how many notes the user got correct
        correct_notes = 0
        beat_offset_list = original_stream.timeSignature.getBeatOffsets()
        beat_offset_list.append(original_stream.timeSignature.barDuration.quarterLength)

        for m in range(min(len(original_stream.getElementsByClass(stream.Measure)), len(user_stream.getElementsByClass(stream.Measure)))):
            for num, b in enumerate(beat_offset_list[:-1]):
                user_notes = [u for u in user_stream.measure(m + 1).getElementsByOffset(b, beat_offset_list[num + 1], 
                                                                            includeEndBoundary=False, mustFinishInSpan=False, 
                                                                            mustBeginInSpan=False, includeElementsThatEndAtStart=False, 
                                                                            classList=(note.Note, note.Rest, chord.Chord)) if 'ChordSymbol' not in u.classSet]
                original_notes = [o for o in original_stream.measure(m + 1).getElementsByOffset(b, beat_offset_list[num + 1],
                                                                                    includeEndBoundary=False, mustFinishInSpan=False,
                                                                                    mustBeginInSpan=False, includeElementsThatEndAtStart=False,
                                                                                    classList=(note.Note, note.Rest, chord.Chord)) if 'ChordSymbol' not in o.classSet]

                for index, original_note in enumerate(original_notes):
                    check_function = correct_chord_check if original_note.isChord else correct_note_check
                    correct_notes += 1 if original_note.offset == b and check_function(original_note, user_notes[index], exact) == True else 0
    
        score = correct_notes / (len(original_notes_list))

    else:
        user_notes = [u for u in user_stream.flatten().getElementsByClass([note.Note, chord.Chord]).stream() if 'ChordSymbol' not in u.classSet]
        original_notes = [o for o in original_stream.flatten().getElementsByClass([note.Note, chord.Chord]).stream() if 'ChordSymbol' not in o.classSet]
        # print(original_notes, user_notes)
        distance, insertions, deletions, substitutions = levenshtein_distance(original_notes, user_notes, exact=exact)
        # print(f"Distance: {distance}, Insertions: {insertions}, Deletions: {deletions}, Substitutions: {substitutions}")
        score = max(0, (len(original_notes_list) - distance) / len(original_notes_list))
    # print(score)
    return True if score >= score_threshold else False

def format_user_stream(user_stream, original_stream, count_in_measures=2, in_time=False):

    if user_stream is None or len([u for u in user_stream.flatten().getElementsByClass([note.Note, chord.Chord]).stream() if 'ChordSymbol' not in u.classSet]) == 0:
        raise ValueError('User stream is empty')

    user_stream = user_stream.parts[0].flat.notesAndRests.stream()

    if in_time:
        original_stream_duration_list = []
        for n in original_stream.flat.notes:
            if n.duration.quarterLength not in original_stream_duration_list:
                original_stream_duration_list.append(1 / n.duration.quarterLength)
        original_stream_duration_list = sorted(original_stream_duration_list, reverse=True)

        user_stream = user_stream.measures(count_in_measures + 1, None) # Delete the first two measures of the user's stream

        user_measure_num = len(user_stream.getElementsByClass(stream.Measure))
        original_measure_num = len(original_stream.getElementsByClass(stream.Measure))
        if user_measure_num > original_measure_num:
            user_stream = user_stream.measures(0, original_measure_num) # Trim the user's stream to the same length as the original excerpt
        elif user_measure_num < original_measure_num:
            for m in range(original_measure_num - user_measure_num):
                user_stream.append(stream.Measure())

        user_stream = user_stream.makeMeasures()
        user_stream = user_stream.quantize(quarterLengthDivisors=tuple(original_stream_duration_list),
                                        processOffsets=True, processDurations=True, recurse=True)
        
        ### Remove any notes that are outside of the time signature of the original excerpt
        for u in user_stream.flatten().getElementsByClass([note.Note, note.Rest, chord.Chord]).stream():
            if 'ChordSymbol' not in u.classSet and u.offset > original_stream.timeSignature.duration.quarterLength:
                user_stream.remove(u)
    
    else:
        user_notes = [u for u in user_stream.flatten().getElementsByClass([note.Note, chord.Chord]).stream() if 'ChordSymbol' not in u.classSet]
        for index, u in enumerate(user_notes):
            if index < (len(user_notes) - 1) and u.offset == user_notes[index + 1].offset:
                user_stream.remove(u if u.tie is None or u.duration.quarterLength < user_notes[index + 1].duration.quarterLength else user_notes[index + 1])
            # Remove tied notes that are not the first in the tie and remove the tie from the first note
            elif u.tie:
                if u.tie.type != 'start':
                    user_stream.remove(u)
                else:
                    u.tie = None

    return user_stream

def correct_note_check(original_note, user_note, exact=True):
    if original_note.isRest and user_note.isRest:
        return False
    elif original_note.isRest and not user_note.isRest or not original_note.isRest and user_note.isRest:
        return False
    elif user_note.pitch.isEnharmonic(original_note.pitch) or user_note.pitch == original_note.pitch:
        return True
    else:
        return False
    
def correct_chord_check(original_chord, user_chord, exact=True):
    if not user_chord.isChord:
        return False
    
    if exact:
        if len(original_chord.pitches) != len(user_chord.pitches):
            return False
        else:
            for index, pitch in enumerate(original_chord.pitches):
                if not user_chord.pitches[index].isEnharmonic(pitch):
                    return False
            return True
    else:
        return True if harmony.chordSymbolFigureFromChord(original_chord) == harmony.chordSymbolFigureFromChord(user_chord) else False
    
### MIDI function
def format_user_midi(user_midi_file, original_stream, bpm, output_filename_midi):
    if user_midi_file is None:
        raise ValueError('User midi file is empty')

    # Set the tempo and time signature of the user's midi file to match the original excerpt
    set_user_tempo = MetaMessage('set_tempo', tempo=bpm2tempo(bpm))
    set_time_signature = MetaMessage('time_signature', numerator=original_stream.flat.timeSignature.numerator, denominator=original_stream.flat.timeSignature.denominator)
    
    for i, msg in enumerate(user_midi_file.tracks[0]):
        if msg.type == 'set_tempo':
            user_midi_file.tracks[0][i] = set_user_tempo
        elif msg.type == 'time_signature':
            user_midi_file.tracks[0][i] = set_time_signature
    
    user_midi_file.save(output_filename_midi)

### Audio function
def process_audio(audio_file, audio_fx, metronome_numpy_array = None):

    def phase_cancel(user_audio_file, metronome_np_array):
        # The audios should be numpy arrays, if they are lists convert them to numpy arrays
        # Check if both audio tracks are of same length, if not, pad the shorter one with zeros
        if len(metronome_np_array) > len(user_audio_file):
            user_audio_file = np.pad(user_audio_file, (0, len(metronome_np_array) - len(user_audio_file)))
        elif len(user_audio_file) > len(metronome_np_array):
            metronome_np_array = np.pad(metronome_np_array, (0, len(user_audio_file) - len(metronome_np_array)))
        
        # Check if the audio tracks are of type int16, if so, convert them to float32
        if metronome_np_array.dtype == np.int16:
            metronome_np_array = metronome_np_array.astype(np.float32)

        # Invert the phase of the metronome audio
        metronome_np_array = -1 * metronome_np_array

        # Add the inverted metronome audio to the user audio
        cancelled_audio = user_audio_file + metronome_np_array

        sf.write("user_recordings/cancelled_audio.wav", cancelled_audio, 44100)

        return cancelled_audio

    if audio_file is None or len(audio_file) == 0:
        raise ValueError('User audio file is empty')
    
    # Read the audio file
    audio, sample_rate = sf.read(audio_file)

    # Check if there's a metronome audio file, if so, phase cancel it
    if metronome_numpy_array is not None:
        pass # I'm still working on this part, I need to find a good way to remove the metronome audio file from the user's recording

    # Compress and normalize the audio file
    compressed_and_normalized = audio_fx(audio, sample_rate)

    # Rename and export the audio file
    new_file_name = audio_file.replace(".wav", "_processed.wav")
    sf.write(new_file_name, compressed_and_normalized, sample_rate)

    return new_file_name

### Recording functions
def record_without_metronome(fs, recording_duration=30): #default recording duration is 30 seconds
    recording = sd.rec(int(recording_duration * fs), samplerate=fs, channels=1)

    if input("Press enter to stop the recording: ") == "":
        sd.stop()  # Stop the recording

    return recording, None

def record_with_metronome(music21_stream, bpm, count_in, fs, score_duration):
    def record_audio(fs, score_duration):
        recording = sd.rec(int(score_duration * fs), samplerate=fs, channels=1)
        time.sleep(score_duration)  # Wait for the recording to finish
        sd.stop()  # Stop the recording
        return recording
    
    if music21_stream is None or len(music21_stream.flatten().getElementsByClass([note.Note, note.Rest]).stream()) == 0:
        raise ValueError('Music21 stream is empty')
    
    metronome_WaveObject, metronome_np_array = generate_metronome_track(music21_stream, bpm, count_in, fs)

    with ThreadPoolExecutor(max_workers=2) as executor:
        recording_future = executor.submit(record_audio, fs, score_duration)
        ### It is really important that the recording and the metronome playing happen at  the exact same time ###
        time.sleep(0.5)  # Wait for the recording to start
        metronome_future = executor.submit(metronome_WaveObject.play)

    user_audio = recording_future.result()  # Get the recording

    return user_audio, metronome_np_array

### Metronome function
def generate_metronome_track(music21_stream, bpm=120, count_in=0, fs=44100):
    def get_bpms_and_measure_numbers(music21_stream, bpm):
        # Get the bpm and measure numbers from the music21 stream
        bpms = [bpm]
        bpms_measure_numbers = [0]

        metronome_marks = music21_stream.getElementsByClass('MetronomeMark')
        if metronome_marks:
            bpms = [b.number for b in metronome_marks]
            measures = music21_stream.getElementsByClass('Measure')
            bpms_measure_numbers = [m.number for m in measures if m.hasElementOfClass('MetronomeMark')]

        return bpms, bpms_measure_numbers

    def make_counts(time_signature):
        # Make a list of counts based on the time signature
        partitions = [int(n[0]) for n in time_signature.beatSequence.partitionDisplay.split('+')]
        if time_signature.denominator == 4 or all(int(i) == 1 for i in partitions):
            partitions = [time_signature.numerator]
        count_list = [[c for c in range(1, i + 1)] for i in partitions]
        return [item for sublist in count_list for item in sublist]

    def calculate_metronome_delay(tempo, time_signature):
        # Calculate the delay between metronome clicks
        beats_per_minute = tempo
        beats_per_second = beats_per_minute / 60

        if time_signature.denominator == 4:  # If the denominator is 4, then it's a simple time signature
            subbeats_per_second = beats_per_second  # For simple time signatures, a beat is a quarter note
        elif time_signature.denominator == 8:  # If the denominator is 8, then it's a compound time signature
            subbeats_per_second = beats_per_second * 3  # For compound time signatures, a beat is a dotted quarter note, which contains three eighth notes

        delay = 1 / subbeats_per_second

        return delay

    if music21_stream is None or len(music21_stream.flatten().getElementsByClass([note.Note, note.Rest]).stream()) == 0:
        raise ValueError('Music21 stream is empty')
    
    # Load metronome sounds
    strong_beat = np.array(AudioSegment.from_wav('metronome_clicks/src_strong_beat.wav').get_array_of_samples())
    sub_strong_beat = np.array(AudioSegment.from_wav('metronome_clicks/src_sub_strong_beat.wav').get_array_of_samples())
    weak_beat = np.array(AudioSegment.from_wav('metronome_clicks/src_weak_beat.wav').get_array_of_samples())

    bpms, bpms_measure_numbers = get_bpms_and_measure_numbers(music21_stream, bpm)
    number_of_measures = len(music21_stream.getElementsByClass('Measure')) + count_in
    time_signature = music21_stream.flat.timeSignature

    metronome_track = []
    for m in range(number_of_measures):
        if m in bpms_measure_numbers:
            bpm = bpms[bpms_measure_numbers.index(m)]
        delay = calculate_metronome_delay(bpm, time_signature)

        for num, c in enumerate(make_counts(time_signature)):
            if c == 1:
                click = strong_beat if num == 0 else sub_strong_beat
            else:
                click = weak_beat
            
            silence_duration = int(delay * fs) - len(click)
            if silence_duration > 0:
                silence = np.zeros(silence_duration)
                metronome_click = np.concatenate((click, silence))
            else:
                metronome_click = click[:int(delay * fs)]

            metronome_track.append(metronome_click)

    metronome_track_numpy_array = np.concatenate(metronome_track).astype(np.int16)
    metronome_track_WaveObject = sa.WaveObject(metronome_track_numpy_array, num_channels=1, bytes_per_sample=2, sample_rate=fs)

    return metronome_track_WaveObject, metronome_track_numpy_array

### Auxiliary functions
def levenshtein_distance(s1, s2, exact=True):
    """Calculates the Levenshtein distance between two sequences and counts insertions, deletions, and substitutions, using correct_note_check and correct_chord_check for comparison."""
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0 for _ in range(len_s2 + 1)] for _ in range(len_s1 + 1)]
    
    for i in range(len_s1 + 1):
        for j in range(len_s2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif (s1[i - 1].isNote and correct_note_check(s1[i - 1], s2[j - 1])) or \
                 (s1[i - 1].isChord and correct_chord_check(s1[i - 1], s2[j - 1], exact=exact)):
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    # Backtrack to find the number of insertions, deletions, and substitutions
    i, j = len_s1, len_s2
    insertions, deletions, substitutions = 0, 0, 0
    
    while i > 0 and j > 0:
        if (s1[i - 1].isNote and correct_note_check(s1[i - 1], s2[j - 1])) or \
           (s1[i - 1].isChord and correct_chord_check(s1[i - 1], s2[j - 1], exact=exact)):
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
    
    insertions += j
    deletions += i
    
    return dp[len_s1][len_s2], insertions, deletions, substitutions

def calculate_quarter_bpm(original_bpm, time_signature):
    # Get the denominator from the time signature
    denominator = time_signature.denominator

    # Calculate the "beat" length (in quarter notes)
    if denominator == 2:
        beat_length = 2  # half note
    elif denominator == 4:
        beat_length = 1  # quarter note
    elif denominator == 8:
        beat_length = 1 / 2  # eighth note
    else:
        beat_length = 1  # default to quarter note (this may not always be appropriate)

    # Now we calculate the new bpm based on quarter notes
    if time_signature.numerator % 3 == 0 and denominator == 8:
        # If we're in a compound meter (top number is multiple of 3 and bottom number is 8), we use dotted quarters
        dotted_quarter_bpm = original_bpm * (2 / 3)  # each dotted quarter contains 3 eighth notes, so we divide by 3 and multiply by 2 (as each dotted quarter is equal to 2 quarter notes)
        return dotted_quarter_bpm
    else:
        # If we're not in a compound meter, we just adjust based on the beat length
        quarter_bpm = original_bpm * beat_length
        return quarter_bpm

def get_score_duration(excerpt, bpm, count_in=0):
    time_signature = excerpt.flat.timeSignature
    number_of_measures = len(excerpt.getElementsByClass(stream.Measure))
    if time_signature.denominator == 8:
        quarter_length = time_signature.numerator / 2
    else:
        quarter_length = time_signature.numerator
    measure_duration = 60 * quarter_length / bpm  # measure duration in seconds
    return measure_duration * (number_of_measures + count_in)  # full score duration in seconds

def get_score_pitch_range(excerpt):
    highest_pitch = 0
    lowest_pitch = 127
    pitch_padding = 5
    for n in excerpt.flatten().getElementsByClass([note.Note, chord.Chord]).stream():
        if n.isNote:
            if n.pitch.midi > highest_pitch:
                highest_pitch = n.pitch.midi
            elif n.pitch.midi < lowest_pitch:
                lowest_pitch = n.pitch.midi
        elif n.isChord and 'ChordSymbol' not in n.classSet:
            sorted_chord = n.sortDiatonicAscending()
            if sorted_chord[-1].pitch.midi > highest_pitch:
                highest_pitch = sorted_chord[-1].pitch.midi
            elif sorted_chord[0].pitch.midi < lowest_pitch:
                lowest_pitch = sorted_chord[0].pitch.midi

    highest_freq = round(pitch.Pitch(midi=highest_pitch + pitch_padding).frequency) if highest_pitch >= 55 else 1000 #Hz
    lowest_freq = round(pitch.Pitch(midi=lowest_pitch - pitch_padding).frequency) if lowest_pitch <= 79 else 200 #Hz

    return highest_freq, lowest_freq

def get_score_min_note_length(excerpt, bpm):
    min_note_length_ms = 999999999
    quarter_length_milliseconds = 60 * 1000 / bpm
    score_notes_and_chords = [n for n in excerpt.flatten().getElementsByClass([note.Note, chord.Chord]).stream() if 'ChordSymbol' not in n.classSet]
    for n in score_notes_and_chords:
        if n.quarterLength * quarter_length_milliseconds < min_note_length_ms:
            min_note_length_ms = n.quarterLength * quarter_length_milliseconds
    return min_note_length_ms
