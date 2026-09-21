"""CLI entry point for Gamelanizer."""

import argparse
import os
import sys

# Ensure package is discoverable when running directly as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.engine.offline import OfflineProcessor
from app.ui.main_window import MainWindow
from app.ui.controls import EngineControls
from app.ui.progress import ProgressBar
from app.ui.visualizer import AsciiVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gamelanizer: Convert Western or acoustic audio into authentic Gamelan music.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=str, default=None, help="Path to input audio file (e.g. guitar.wav)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path for rendered Gamelan WAV output")
    parser.add_argument("-m", "--midi", type=str, default=None, help="Path for optional output MIDI file")
    parser.add_argument(
        "-s", "--scale",
        type=str,
        choices=["slendro", "pelog", "diatonic"],
        default="slendro",
        help="Target Gamelan tuning system (slendro, pelog, or diatonic 12-TET hybrid)",
    )
    parser.add_argument(
        "--pathet",
        type=str,
        default=None,
        help="Gamelan modal pathet (e.g., slendro_nem, pelog_barang)",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        choices=["saron", "demung", "peking", "bonang", "gong"],
        default="saron",
        help="Gamelan lead instrument to synthesize",
    )
    parser.add_argument("--quantize", action="store_true", default=True, help="Quantize note onsets to rhythmic grid")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false", help="Disable rhythmic quantization")
    parser.add_argument("--transpose", type=float, default=0.0, help="Manual transposition shift in semitones (e.g. -2, 1)")
    parser.add_argument("--auto-key", action="store_true", default=True, help="Automatically align song key with Gamelan scale")
    parser.add_argument("--no-auto-key", dest="auto_key", action="store_false", help="Disable auto-key alignment")
    parser.add_argument("--legato", action="store_true", default=True, help="Enable natural ringing decay and note overlap")
    parser.add_argument("--no-legato", dest="legato", action="store_false", help="Disable legato mode")
    parser.add_argument("--reverb", action="store_true", default=True, help="Apply acoustic Pendopo reverb ambiance")
    parser.add_argument("--no-reverb", dest="reverb", action="store_false", help="Disable Pendopo reverb")
    parser.add_argument("--gong", action="store_true", default=True, help="Generate automatic Gong Ageng & Kenong punctuation layer")
    parser.add_argument("--no-gong", dest="gong", action="store_false", help="Disable Gong accompaniment layer")
    parser.add_argument("--bonang", action="store_true", default=True, help="Generate automatic Bonang Barung chime embellishment layer")
    parser.add_argument("--no-bonang", dest="bonang", action="store_false", help="Disable Bonang embellishment layer")
    parser.add_argument("--stereo", action="store_true", default=True, help="Render in immersive stereo width")
    parser.add_argument("--mono", dest="stereo", action="store_false", help="Render in mono")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal wizard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.interactive or args.input is None:
        ctrls = EngineControls(
            scale=args.scale,
            pathet=args.pathet or "",
            instrument=args.instrument,
            quantize=args.quantize,
            auto_key=args.auto_key,
            transpose=args.transpose,
            legato=args.legato,
            reverb=args.reverb,
            add_gong=args.gong,
            add_bonang=args.bonang,
            stereo=args.stereo,
        )
        window = MainWindow(controls=ctrls)
        window.run_cli_interactive()
        return

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[Error] Input audio file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    output_wav = args.output
    if not output_wav:
        os.makedirs("output", exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_wav = os.path.join("output", f"{base}_{args.scale}_{args.instrument}.wav")

    print("=" * 65)
    print("                      GAMELANIZER CLI                      ")
    print("=" * 65)
    print(f"Input file  : {input_path}")
    print(f"Tuning scale: {args.scale.upper()}")
    print(f"Lead Inst   : {args.instrument.upper()}")
    print(f"Quantization: {'ON' if args.quantize else 'OFF'}")
    key_mode = f"AUTO ({'Enabled' if args.auto_key else 'Disabled'})"
    if args.transpose != 0.0:
        key_mode += f" [Manual Shift: {args.transpose:+.1f} st]"
    print(f"Key Mode    : {key_mode}")
    print(f"Acoustics   : {'Physical Modeling + Legato' if args.legato else 'Physical Modeling (Damped)'} | {'Pendopo Reverb (ON)' if args.reverb else 'Reverb (OFF)'} | {'Stereo' if args.stereo else 'Mono'}")
    print(f"Ensemble    : Bonang Layer: {'ON' if args.bonang else 'OFF'} | Gong Ageng: {'ON' if args.gong else 'OFF'}")
    print(f"Output WAV  : {output_wav}")
    if args.midi:
        print(f"Output MIDI : {args.midi}")
    print("-" * 65)

    pbar = ProgressBar(description="Converting")
    processor = OfflineProcessor(
        scale_name=args.scale,
        pathet=args.pathet,
        instrument_name=args.instrument,
        quantize=args.quantize,
        auto_key=args.auto_key,
        transpose=args.transpose,
        legato=args.legato,
        reverb=args.reverb,
        add_gong=args.gong,
        add_bonang=args.bonang,
        stereo=args.stereo,
    )

    try:
        result = processor.process_file(
            input_path=input_path,
            output_wav_path=output_wav,
            output_midi_path=args.midi,
            progress_cb=pbar.update,
        )

        print("\n" + "=" * 65)
        print("CONVERSION SUCCESSFUL!")
        print("=" * 65)
        print(f"Total Lead Notes Transcribed : {len(result.detected_notes)}")
        if result.transposition_applied != 0.0:
            print(f"Harmonic Transposition       : {result.transposition_applied:+.1f} semitones")
        if result.bonang_notes:
            print(f"Bonang Chime Ensemble Layer  : {len(result.bonang_notes)} interlocking notes")
        if result.gong_notes:
            print(f"Gong Punctuation Layer       : {len(result.gong_notes)} structural strokes (Gong & Kenong)")
        print(f"Rendered Audio File          : {os.path.abspath(output_wav)}")

        AsciiVisualizer.print_note_table(result.mapped_notes, max_notes=12)
        AsciiVisualizer.print_piano_roll(result.mapped_notes)

    except Exception as e:
        print(f"\n[Error executing pipeline] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
