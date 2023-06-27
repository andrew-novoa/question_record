import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import simpleaudio as sa
import sounddevice as sd
import soundfile as sf
from basic_pitch.inference import predict
from mido import MetaMessage, MidiFile, bpm2tempo
from music21 import converter, note, pitch, stream
from pedalboard import Compressor, Gain, Limiter, Pedalboard
from pydub import AudioSegment
from scipy.io.wavfile import write

from qa_generate.excerpt_test_2 import generate_excerpt

### Main function
def record_and_grade(in_time=True, fs=44100, excerpt=None, bpm=120, count_in=2):

    quarter_bpm = calculate_quarter_bpm(bpm, excerpt.timeSignature) # Calculate the quarter note BPM for the MIDI file
    full_duration = get_score_duration(excerpt, bpm, count_in=count_in)  # seconds
    highest_freq, lowest_freq = get_score_pitch_range(excerpt) # Hz
    min_note_length_ms = get_score_min_note_length(excerpt, bpm) # milliseconds
    board = Pedalboard([Compressor(threshold_db=-20, ratio=8, attack_ms=10, release_ms=min_note_length_ms), 
                        Gain(gain_db=8), Limiter(threshold_db=-10, release_ms=min_note_length_ms)])
    
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
                                                   frame_threshold=0.2 if in_time else 0.5, 
                                                   midi_tempo=quarter_bpm if in_time else 120,
                                                   maximum_frequency=highest_freq,
                                                   minimum_frequency=lowest_freq,
                                                   minimum_note_length=min_note_length_ms if in_time else 50,
                                                   melodia_trick=True
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
    return compare_m21_streams(excerpt, user_m21_stream_formatted, in_time)

### Music21 functions
def compare_m21_streams(original_stream, user_stream, in_time=False):
    def correct_note_check(original_note, user_note):
        if original_note.isRest and user_note.isRest:
            return False
        elif original_note.isRest and not user_note.isRest or not original_note.isRest and user_note.isRest:
            return False
        elif user_note.pitch.isEnharmonic(original_notes[index].pitch):
            return True

    if in_time:
        ### Iterate over each beat in the original stream and determine a percentage-based score based on how many notes the user got correct
        correct_notes = 0
        beat_offset_list = original_stream.timeSignature.getBeatOffsets()
        beat_offset_list.append(original_stream.timeSignature.barDuration.quarterLength)

        for m in range(len(original_stream.getElementsByClass(stream.Measure))):
            for num, b in enumerate(beat_offset_list[:-1]):
                user_notes = [u for u in user_stream.measure(m + 1).getElementsByOffset(b, beat_offset_list[num + 1], 
                                                                            includeEndBoundary=False, mustFinishInSpan=False, 
                                                                            mustBeginInSpan=False, includeElementsThatEndAtStart=False, 
                                                                            classList=(note.Note, note.Rest))]
                original_notes = [o for o in original_stream.measure(m + 1).getElementsByOffset(b, beat_offset_list[num + 1],
                                                                                    includeEndBoundary=False, mustFinishInSpan=False,
                                                                                    mustBeginInSpan=False, includeElementsThatEndAtStart=False,
                                                                                    classList=(note.Note, note.Rest))]

                for index, original_note in enumerate(original_notes):
                    correct_notes += 1 if original_note.offset == b and correct_note_check(original_note, user_notes[index]) == True else 0
                    
        score = correct_notes / (len(original_stream.flatten().getElementsByClass(note.Note).stream()))

    else:
        wrong_notes = 0
        original_notes = original_stream.flatten().getElementsByClass(note.Note).stream()
        user_notes = user_stream.flatten().getElementsByClass(note.Note).stream()
        for index, s in enumerate(original_notes):
            if len(user_notes) < index:
                break
            if user_notes[index].tie is not None:
                user_notes.pop(index)
            while s.pitch.isEnharmonic(user_notes[index].pitch) == False:
                if len(user_notes) < index:
                    break
                user_notes.pop(index)
                wrong_notes += 1
        score = (len(original_notes) - wrong_notes) / len(original_notes)

    # print(f"Score: {round(score * 100, 2)}")
    return True if score >= 0.8 else False

def format_user_stream(user_stream, original_stream, count_in_measures=2, in_time=False):
    user_stream = user_stream.parts[0]

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
        for u in user_stream.flatten().getElementsByClass([note.Note, note.Rest]):
            if u.offset > original_stream.timeSignature.duration.quarterLength:
                user_stream.remove(u)
    
    else:
        user_notes = user_stream.flatten().getElementsByClass(note.Note)
        for index, u in enumerate(user_notes):
            if index < (len(user_notes) - 1) and u.offset == user_notes[index + 1].offset:
                user_stream.remove(u if u.tie is None or u.duration.quarterLength < user_notes[index + 1].duration.quarterLength else user_notes[index + 1])
        
    user_stream = user_stream.makeNotation()

    return user_stream

### MIDI function
def format_user_midi(user_midi_file, original_stream, bpm, output_filename_midi):
    # Set the tempo and time signature of the user's midi file to match the original excerpt
    set_user_tempo = MetaMessage('set_tempo', tempo=bpm2tempo(bpm))
    set_time_signature = MetaMessage('time_signature', numerator=original_stream.timeSignature.numerator, denominator=original_stream.timeSignature.denominator)
    
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

    # Load metronome sounds
    strong_beat = np.array(AudioSegment.from_wav('metronome_clicks/src_strong_beat.wav').get_array_of_samples())
    sub_strong_beat = np.array(AudioSegment.from_wav('metronome_clicks/src_sub_strong_beat.wav').get_array_of_samples())
    weak_beat = np.array(AudioSegment.from_wav('metronome_clicks/src_weak_beat.wav').get_array_of_samples())

    bpms, bpms_measure_numbers = get_bpms_and_measure_numbers(music21_stream, bpm)
    number_of_measures = len(music21_stream.getElementsByClass('Measure')) + count_in
    time_signature = music21_stream.timeSignature

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
    number_of_measures = len(excerpt.getElementsByClass(stream.Measure))
    if excerpt.timeSignature.denominator == 8:
        quarter_length = excerpt.timeSignature.numerator / 2
    else:
        quarter_length = excerpt.timeSignature.numerator
    measure_duration = 60 * quarter_length / bpm  # measure duration in seconds
    return measure_duration * (number_of_measures + count_in)  # full score duration in seconds

def get_score_pitch_range(excerpt):
    highest_pitch = 0
    lowest_pitch = 127
    pitch_padding = 5
    for n in excerpt.flatten().getElementsByClass(note.Note):
        if n.pitch.midi > highest_pitch:
            highest_pitch = n.pitch.midi
        elif n.pitch.midi < lowest_pitch:
            lowest_pitch = n.pitch.midi

    highest_freq = round(pitch.Pitch(midi=highest_pitch + pitch_padding).frequency)
    lowest_freq = round(pitch.Pitch(midi=lowest_pitch - pitch_padding).frequency)

    return highest_freq, lowest_freq

def get_score_min_note_length(excerpt, bpm):
    min_note_length_ms = 999999999
    quarter_length_milliseconds = 60 * 1000 / bpm
    for n in excerpt.flatten().getElementsByClass(note.Note):
        if n.quarterLength * quarter_length_milliseconds < min_note_length_ms:
            min_note_length_ms = n.quarterLength * quarter_length_milliseconds
    return min_note_length_ms

### Import the excerpt
excerpt = generate_excerpt(input_time_sig=[[4],[4]], input_key_sig=[random.choice([-1, 0, 1])], input_length=3, input_subdivision=4) # Generate a random excerpt (this is an updated function from excerpt_test_2 that I will add to the previous repos soon)
excerpt.show()

# Record and grade the user's performance
if input("Press enter to start recording:") == '':
    print(record_and_grade(in_time=True, excerpt=excerpt, bpm=100, count_in=2))
