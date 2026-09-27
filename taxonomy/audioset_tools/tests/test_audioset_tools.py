from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from audioset_tools import (
    AUDIOSET,
    AUDIOSET_STRONG,
    VGGSOUND,
    AudioDataset,
    DownloadJob,
    SaltResolver,
    YouTubeDatasetDownloader,
    detect_format,
    generate_event_tracks,
)


ROOT = Path(__file__).resolve().parents[3]


class AudioSetToolsTest(unittest.TestCase):
    def test_detects_supported_local_metadata_formats(self):
        self.assertEqual(
            detect_format(ROOT / "datasets/AudioSet_meta/eval_segments.csv"),
            AUDIOSET,
        )
        self.assertEqual(
            detect_format(ROOT / "datasets/AudioSet_Strong_meta/audioset_eval_strong.tsv"),
            AUDIOSET_STRONG,
        )
        self.assertEqual(
            detect_format(ROOT / "datasets/VGGsound_meta/vggsound.csv"),
            VGGSOUND,
        )

    def test_audioset_parser_preserves_multiple_labels_and_resolves_salt(self):
        salt = SaltResolver.from_mapping_file()
        dataset = AudioDataset.from_file(
            ROOT / "datasets/AudioSet_meta/eval_segments.csv",
            labels_file=ROOT / "datasets/AudioSet_meta/class_labels_indices.csv",
            salt_resolver=salt,
        )
        first = dataset[0]
        self.assertEqual(first.youtube_id, "--4gqARaEJE")
        self.assertEqual(
            first.original_label_ids,
            ("/m/068hy", "/m/07q6cd_", "/m/0bt9lr", "/m/0jbk"),
        )
        self.assertIn("Domestic animals, pets", first.original_labels)
        self.assertIn("animal", first.salt_labels)

    def test_strong_parser_groups_events_by_segment_id(self):
        salt = SaltResolver.from_mapping_file()
        dataset = AudioDataset.from_file(
            ROOT / "datasets/AudioSet_Strong_meta/audioset_eval_strong.tsv",
            labels_file=ROOT / "datasets/AudioSet_Strong_meta/mid_to_display_name.tsv",
            salt_resolver=salt,
        )
        first = dataset[0]
        self.assertEqual(first.source_id, "s9d-2nhuJCQ_30000")
        self.assertEqual(first.youtube_id, "s9d-2nhuJCQ")
        self.assertEqual(first.start_seconds, 30.0)
        self.assertEqual(len(first.events), 6)

    def test_vggsound_parser_and_salt_filtering(self):
        salt = SaltResolver.from_mapping_file()
        dataset = AudioDataset.from_file(
            ROOT / "datasets/VGGsound_meta/vggsound.csv",
            salt_resolver=salt,
        )
        first = dataset[0]
        self.assertEqual(first.youtube_id, "---g-f_I2yQ")
        self.assertEqual(first.start_seconds, 1.0)
        self.assertEqual(first.end_seconds, 11.0)
        self.assertEqual(first.split, "test")
        filtered = dataset.filter_labels(
            ["vehicle"],
            label_space="salt",
            include_descendants=True,
        )
        self.assertGreater(len(filtered), 0)

    def test_event_tracks_are_generated_from_normalized_strong_records(self):
        dataset = AudioDataset.from_file(
            ROOT / "datasets/AudioSet_Strong_meta/audioset_eval_strong.tsv",
            labels_file=ROOT / "datasets/AudioSet_Strong_meta/mid_to_display_name.tsv",
        )
        tracks = generate_event_tracks(dataset.select_range(0, 1), bin_size=0.5)
        self.assertTrue(tracks)
        self.assertEqual(tracks[0][1].shape, (20,))

    def test_downloader_dry_run_and_invalid_youtube_id_report(self):
        valid_job = DownloadJob(
            dataset=VGGSOUND,
            source_id="---g-f_I2yQ_1",
            youtube_id="---g-f_I2yQ",
            start_seconds=1.0,
            end_seconds=11.0,
        )
        with TemporaryDirectory() as tmp:
            downloader = YouTubeDatasetDownloader(tmp, dry_run=True)
            valid_report = downloader.download_job(valid_job)
            self.assertTrue(valid_report.success)
            self.assertEqual(valid_report.status, "skipped_dry_run")

            invalid_report = downloader.download_job(
                DownloadJob(dataset=VGGSOUND, source_id="bad", youtube_id="not-valid")
            )
            self.assertFalse(invalid_report.success)
            self.assertEqual(invalid_report.error_category, "InvalidYouTubeId")


if __name__ == "__main__":
    unittest.main()
