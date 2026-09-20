"""Visualizer: piano-roll style display and pitch mapping comparisons."""

from typing import List
from app.music.note import Note


class AsciiVisualizer:
    """Renders formatted console tables and piano-roll comparisons of notes."""

    @staticmethod
    def print_note_table(notes: List[Note], max_notes: int = 20) -> None:
        """Print a formatted ASCII table comparing detected vs Gamelan pitches."""
        print("-" * 75)
        print(f"{'#':<3} | {'Start (s)':<9} | {'Dur (s)':<7} | {'Orig Hz':<8} | {'MIDI':<6} | {'Gamelan':<8} | {'Tuned Hz':<8}")
        print("-" * 75)

        for i, n in enumerate(notes[:max_notes]):
            solf = n.metadata.get("gamelan_solfege", "")
            gam_str = f"{n.gamelan_note} ({solf})" if n.gamelan_note else "-"
            tuned_hz = f"{n.gamelan_freq_hz:.1f}" if n.gamelan_freq_hz else "-"
            print(
                f"{i+1:<3} | {n.start_time:<9.2f} | {n.duration:<7.2f} | "
                f"{n.pitch_hz:<8.1f} | {n.western_name:<6} | {gam_str:<8} | {tuned_hz:<8}"
            )

        if len(notes) > max_notes:
            print(f"... and {len(notes) - max_notes} more notes.")
        print("-" * 75)

    @staticmethod
    def print_piano_roll(notes: List[Note], total_width: int = 60) -> None:
        """Simple ASCII piano roll representation over time."""
        if not notes:
            print("[Empty notes sequence]")
            return

        total_dur = max(n.end_time for n in notes)
        if total_dur <= 0:
            return

        print("\n--- Gamelan Note Timeline ---")
        gamelan_degrees = ["1'", "6", "5", "3", "2", "1"]
        grid = {deg: [" "] * total_width for deg in gamelan_degrees}

        for n in notes:
            deg = n.gamelan_note or "1"
            if deg not in grid:
                deg = "1"
            start_col = min(total_width - 1, int((n.start_time / total_dur) * total_width))
            end_col = min(total_width, max(start_col + 1, int((n.end_time / total_dur) * total_width)))
            for c in range(start_col, end_col):
                grid[deg][c] = "#"

        for deg in gamelan_degrees:
            row_str = "".join(grid[deg])
            print(f"Laras {deg:<2} |{row_str}|")
        print(f"{'Time':<9} 0s" + " " * (total_width - 7) + f"{total_dur:.1f}s\n")
