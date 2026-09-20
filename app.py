from flask import Flask, render_template, request, send_file
import os
import subprocess
import librosa
import pretty_midi
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Fungsi mengubah audio drum menjadi file MIDI (deteksi ketukan)
def audio_to_midi(audio_path, output_midi_path):
    y, sr = librosa.load(audio_path)
    # Deteksi ketukan (onset)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    midi_data = pretty_midi.PrettyMIDI()
    gamelan_program = pretty_midi.instrument_name_to_program('Glockenspiel') # Mendekati gamelan saron jika metadata sf2 tidak ada
    gamelan_track = pretty_midi.Instrument(program=gamelan_program, is_drum=False)
    
    # Buat nada setiap kali ada ketukan drum
    for time in onset_times:
        note = pretty_midi.Note(
            velocity=100, pitch=60, start=time, end=time + 0.2
        )
        gamelan_track.notes.append(note)
        
    midi_data.instruments.append(gamelan_track)
    midi_data.write(output_midi_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return "Tidak ada file", 400
        
    file = request.files['file']
    if file.filename == '':
        return "Tidak ada file yang dipilih", 400
        
    # Beri nama unik agar tidak bentrok jika diakses banyak orang
    job_id = str(uuid.uuid4())
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.wav")
    file.save(input_path)
    
    # Tahap 1: Pisahkan lagu pakai Demucs untuk ambil Drum-nya saja
    subprocess.run(f"demucs --two-stems=drums -o {app.config['OUTPUT_FOLDER']} {input_path}", shell=True)
    
    # Ambil file hasil ekstrak drum
    drum_audio_path = f"{app.config['OUTPUT_FOLDER']}/htdemucs/{job_id}/drums.wav"
    midi_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}.mid")
    final_output = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_gamelan.wav")
    
    try:
        # Tahap 2: Ubah Drum menjadi notasi MIDI (ketukan)
        audio_to_midi(drum_audio_path, midi_path)
        
        # Tahap 3: Render MIDI dengan SoundFont Gamelan
        soundfont = "gamelan.sf2"
        subprocess.run(f"fluidsynth -ni -F {final_output} {soundfont} {midi_path}", shell=True)
        
        # Kembalikan file hasil ke website untuk didownload/diputar
        return send_file(final_output, as_attachment=False)
        
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    # Jalankan website di port 5000
    app.run(debug=True, port=5000)