# Gamelanizer 🎵🇮🇩

**Gamelanizer** adalah framework pemrosesan audio berbasis Python untuk mengubah instrumen barat/akustik (seperti gitar, vokal, atau keyboard) menjadi komposisi musik Gamelan tradisional Jawa dan Bali yang autentik.

Sistem mendeteksi pitch dan transient ketukan (onset), memetakan frekuensi ke tangga nada **Laras Slendro** atau **Laras Pelog** secara microtonal, serta mensintesis suara bilah perunggu/besi instrumen Gamelan (seperti Saron Barung, Demung, Peking, Bonang, dan Gong) menggunakan *modal physical synthesis* dan *sample playback*.

---

## 📁 Struktur Proyek

```text
gamelanizer/
│
├── app/
│   ├── __init__.py
│   ├── main.py                         # CLI entry point utama
│   │
│   ├── audio/                          # Modul Audio I/O & Analisis
│   │   ├── __init__.py
│   │   ├── loader.py                   # Loading & saving WAV/audio
│   │   ├── preprocessing.py            # Normalisasi & filter
│   │   └── analyzer.py                 # Ekstraksi durasi, RMS & tempo
│   │
│   ├── pitch/                          # Modul Deteksi Nada
│   │   ├── __init__.py
│   │   ├── base.py                     # BasePitchDetector interface
│   │   ├── pyin_detector.py            # Algoritma pYIN & autocorrelation
│   │   ├── postprocess.py              # Smoothing & segmentasi notasi
│   │   └── onset.py                    # Deteksi ketukan transient
│   │
│   ├── music/                          # Modul Notasi Musik
│   │   ├── __init__.py
│   │   ├── note.py                     # Struktur data Note
│   │   ├── events.py                   # NoteEvent, Track & Score
│   │   ├── timing.py                   # Grid ketukan & tempo
│   │   ├── velocity.py                 # Estimasi dinamika pukulan (1-127)
│   │   ├── tuning.py                   # Formula tala 12-TET Western
│   │   └── quantizer.py                # Kuantisasi ritmis
│   │
│   ├── gamelan/                        # Modul Domain Gamelan
│   │   ├── __init__.py
│   │   ├── tuning.py                   # Tabel tala Slendro & Pelog
│   │   ├── scale.py                    # Tangga nada & Pathet modes
│   │   ├── instrument.py               # Profil Saron, Demung, Peking, dll.
│   │   ├── mapping.py                  # Pemetaan microtonal pitch
│   │   ├── performance.py              # Gaya tabuhan & teknik peredaman (mathet)
│   │   ├── sampler.py                  # Modal physical sound synthesizer
│   │   ├── sample_manager.py           # Manajemen sampel WAV pada disk
│   │   ├── envelope.py                 # ADSR decay bilah logam
│   │   └── renderer.py                 # Synthesizer audio track Gamelan
│   │
│   ├── engine/                         # Core Execution Engine
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # End-to-end processing pipeline
│   │   ├── offline.py                  # Konversi batch file-to-file
│   │   ├── renderer.py                 # Master bus export (stereo/gain/limiter)
│   │   ├── realtime.py                 # Pemrosesan streaming real-time
│   │   ├── audio_thread.py             # Thread worker asinkron
│   │   ├── event_queue.py              # Queue thread-safe event
│   │   └── latency.py                  # Pengukuran latensi buffer
│   │
│   └── ui/                             # Antarmuka Pengguna
│       ├── __init__.py
│       ├── main_window.py              # UI controller interaktif
│       ├── file_selector.py            # File dialog & validasi
│       ├── controls.py                 # Pengaturan parameter konversi
│       ├── progress.py                 # Indikator progres CLI/GUI
│       └── visualizer.py               # Visualisasi ASCII timeline notasi
│
├── data/
│   ├── input/
│   │   └── guitar.wav                  # Audio sample uji coba
│   │
│   ├── samples/
│   │   └── saron/
│   │       ├── metadata.json           # Metadata sample bilah Saron
│   │       └── ...
│   │
│   └── metadata/
│       └── tuning/
│           ├── slendro.json            # Referensi tala Slendro
│           └── pelog.json              # Referensi tala Pelog
│
├── output/                             # Direktori hasil render audio/midi
│
├── tests/                              # Unit Test Suite
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_pitch.py
│   ├── test_music.py
│   ├── test_gamelan.py
│   └── test_engine.py
│
├── scripts/
│   └── generate_sample_audio.py        # Generator audio demo sintetis
│
├── notebooks/                          # Jupyter notebooks riset/eksperimen
│
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Instalasi & Persiapan

1. Pastikan Python 3.10+ telah terpasang.
2. Pasang dependensi yang dibutuhkan:
```bash
pip install -r requirements.txt
```

---

## 🎹 Cara Penggunaan

### 1. Menjalankan Konversi via CLI
Mengonversi audio gitar input menjadi nada Gamelan Saron berlaras **Slendro**:
```bash
python app/main.py --input data/input/guitar.wav --scale slendro --instrument saron
```

Mengonversi menggunakan laras **Pelog** dengan instrumen **Demung** dan ekspor MIDI:
```bash
python app/main.py --input data/input/guitar.wav --scale pelog --instrument demung --output output/my_pelog.wav --midi output/my_pelog.mid
```

### 2. Mode Terminal Interaktif
Jalankan tanpa argumen atau dengan flag `--interactive` untuk panduan langkah demi langkah:
```bash
python app/main.py --interactive
```

---

## 🧪 Menjalankan Pengujian (Unit Tests)

Jalankan seluruh suite pengujian otomatis menggunakan `pytest`:
```bash
pytest tests/ -v
```

---

## 🎼 Sistem Tala (Tuning System)

- **Laras Slendro**: Sistem pentatonik berjarak relatif rata (~240 sen per interval) dengan nada:
  - 1 (Ji / Panunggal) ~ 270 Hz
  - 2 (Ro / Gulu) ~ 310 Hz
  - 3 (Lu / Dhadha) ~ 355 Hz
  - 5 (Ma / Lima) ~ 410 Hz
  - 6 (Nem / Nem) ~ 470 Hz
- **Laras Pelog**: Sistem heptatonik dengan kombinasi interval sempit dan lebar:
  - 1 (Ji), 2 (Ro), 3 (Lu), 4 (Pat), 5 (Ma), 6 (Nem), 7 (Pi / Barang)

---

## 📜 Lisensi
MIT License - Gamelanizer Open Source Project.
