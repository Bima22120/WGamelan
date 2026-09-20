"""Main UI controller: runs interactive CLI or GUI workspace."""

import os
import sys
from app.ui.controls import EngineControls
from app.ui.file_selector import FileSelector
from app.ui.progress import ProgressBar
from app.ui.visualizer import AsciiVisualizer
from app.engine.offline import OfflineProcessor


class MainWindow:
    """Coordinates UI interactions and links user inputs with the Gamelanizer engine."""

    def __init__(self, controls: EngineControls = None):
        self.controls = controls or EngineControls()

    def run_cli_interactive(self) -> None:
        """Run interactive terminal interface."""
        print("=" * 60)
        print("          GAMELANIZER - Audio to Gamelan Converter          ")
        print("=" * 60)

        # 1. Select input file
        input_path = input("Enter path to audio file (or press Enter to browse): ").strip()
        if not input_path:
            browse_path = FileSelector.browse_open_file()
            if browse_path:
                input_path = browse_path
            else:
                input_path = "data/input/guitar.wav"
                print(f"Using default: {input_path}")

        if not os.path.exists(input_path):
            print(f"[Error] File not found: {input_path}")
            return

        # 2. Select Scale
        scale_choice = input(f"Select tuning scale [1=Slendro, 2=Pelog] (current: {self.controls.scale}): ").strip()
        if scale_choice == "2":
            self.controls.scale = "pelog"
        elif scale_choice == "1":
            self.controls.scale = "slendro"

        # 3. Select Instrument
        inst_choice = input(f"Select instrument [saron/demung/peking/bonang/gong] (current: {self.controls.instrument}): ").strip().lower()
        if inst_choice in ["saron", "demung", "peking", "bonang", "gong"]:
            self.controls.instrument = inst_choice

        # Output paths
        output_wav = FileSelector.get_default_output_path(input_path, suffix=f"_{self.controls.scale}_{self.controls.instrument}.wav")
        output_mid = output_wav.replace(".wav", ".mid") if self.controls.export_midi else None

        print(f"\nProcessing: {input_path}")
        print(f"Target Scale: {self.controls.scale.upper()} | Instrument: {self.controls.instrument.upper()}")
        print(f"Output Audio: {output_wav}")

        # 4. Run Conversion
        pbar = ProgressBar(total_steps=100, description="Gamelanizing")
        processor = OfflineProcessor(
            scale_name=self.controls.scale,
            pathet=self.controls.pathet or None,
            instrument_name=self.controls.instrument,
            sample_rate=self.controls.sample_rate,
            quantize=self.controls.quantize,
            auto_key=self.controls.auto_key,
            transpose=self.controls.transpose,
        )

        try:
            result = processor.process_file(
                input_path=input_path,
                output_wav_path=output_wav,
                output_midi_path=output_mid,
                progress_cb=pbar.update,
            )

            print("\n" + "=" * 60)
            print("CONVERSION FINISHED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Notes detected: {len(result.detected_notes)}")
            print(f"Total audio duration: {result.summary.duration_sec:.2f}s")
            print(f"Output saved to: {output_wav}")
            if output_mid and os.path.exists(output_mid):
                print(f"Output MIDI saved to: {output_mid}")

            # 5. Visualize notes
            AsciiVisualizer.print_note_table(result.mapped_notes, max_notes=15)
            AsciiVisualizer.print_piano_roll(result.mapped_notes)

        except Exception as e:
            print(f"\n[Error during processing] {e}")
