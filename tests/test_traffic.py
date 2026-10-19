import unittest
import json
from pathlib import Path

from scripts.prepare_map import prepare_signals
from src.map.json_loader import load_trafficlights
from src.simulation.traffic import signal_phase


class SignalPhaseTests(unittest.TestCase):
    def test_changes_exactly_at_phase_boundary_and_repeats(self):
        phases = [{"state": "red", "duration_s": 5}, {"state": "green", "duration_s": 8}]
        self.assertEqual(signal_phase(phases, 0), ("red", 5))
        self.assertEqual(signal_phase(phases, 4.2), ("red", 1))
        self.assertEqual(signal_phase(phases, 5), ("green", 8))
        self.assertEqual(signal_phase(phases, 13), ("red", 5))

    def test_different_programs_have_independent_timers(self):
        first = [{"state": "red", "duration_s": 5}, {"state": "green", "duration_s": 8}]
        second = [{"state": "red", "duration_s": 9}, {"state": "green", "duration_s": 6}]
        self.assertEqual(signal_phase(first, 6), ("green", 7))
        self.assertEqual(signal_phase(second, 6), ("red", 3))


class SignalMapTests(unittest.TestCase):
    def test_only_programmed_signals_are_prepared_without_directions(self):
        class Projection:
            def project(self, lat, lon):
                return lon, lat

        nodes = {
            1: {"lat": 2.0, "lon": 3.0, "tags": {"highway": "traffic_signals"}},
            2: {"lat": 4.0, "lon": 5.0, "tags": {"highway": "traffic_signals"}},
            3: {"lat": 6.0, "lon": 7.0, "tags": {"highway": "traffic_signals"}},
        }
        phases = [{"state": "red", "duration_s": 5}]
        signals = prepare_signals(nodes, Projection(), {1: phases, 2: phases})
        self.assertEqual([signal["osm_id"] for signal in signals], [1, 2])
        self.assertTrue(all("controlled_approach_edge_ids" not in signal for signal in signals))

    def test_prepared_map_contains_only_programmed_signals_without_directions(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "data/prepared/map_data.json").read_text(encoding="utf-8"))
        programmed_ids = set(load_trafficlights(root / "data/trafficlights.yaml"))
        self.assertEqual({signal["osm_id"] for signal in data["signals"]}, programmed_ids)
        for signal in data["signals"]:
            self.assertNotIn("controlled_approach_edge_ids", signal)
            self.assertTrue(signal["is_programmed"])


if __name__ == "__main__":
    unittest.main()
