"""Run existing prediction with a selected four-minute chant sequence."""

import json
import argparse
import runpy
import sys
from pathlib import Path

import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Choose the chant order, then run existing prediction.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--order",
        required=True,
        choices=["YA_HA", "HA_YA"],
        help="YA_HA: Silence/Ya/Silence/Ha; HA_YA: Silence/Ha/Silence/Ya",
    )

    # Forward the remaining arguments to predict_videos.py.
    args, prediction_args = parser.parse_known_args()

    labels = {
        "YA_HA": ["silence", "Ya", "silence", "Ha"],
        "HA_YA": ["silence", "Ha", "silence", "Ya"],
    }[args.order]

    original_analyse_inputs = pipeline.analyse_inputs

    def analyse_first_four_minutes(cfg, model):
        cfg = dict(cfg)
        cfg["duration_tolerance_seconds"] = 2.0
        cfg["max_input_seconds"] = 240.0
        cfg['protocol']={'segments':[dict(start=i*60,end=(i+1)*60,label=label,pair=i//2+1) for i,label in enumerate(labels)]}
        metadata=Path(cfg['output_path'])/'prediction_metadata.json'
        if metadata.exists():
            data=json.loads(metadata.read_text());data['protocol']=cfg['protocol']['segments'];metadata.write_text(json.dumps(data,indent=2))
        (Path(cfg['output_path'])/'prediction_config.yaml').write_text(__import__('yaml').safe_dump(cfg))
        return original_analyse_inputs(cfg, model)

    pipeline.analyse_inputs = analyse_first_four_minutes
    original_protocol = pipeline.protocol

    def selected_protocol(duration, settings):
        # Preserve the existing timing and pairing logic.
        segments = original_protocol(duration, settings)

        if len(segments) != 4:
            raise ValueError(
                "Expected four protocol segments. "
                "Check the existing prediction configuration."
            )

        expected_times = [
            (0, 60),
            (60, 120),
            (120, 180),
            (180, 240),
        ]

        updated = []
        for index, (segment, label, expected) in enumerate(
            zip(segments, labels, expected_times)
        ):
            start, end = expected

            if (
                abs(segment["start"] - start) > 0.01
                or abs(segment["end"] - end) > 1.01
            ):
                raise ValueError(
                    "Expected four consecutive 60-second segments "
                    "starting at 0 seconds. Check the video and protocol."
                )

            item = dict(segment)
            item["label"] = label
            item["pair"] = index // 2 + 1
            updated.append(item)

        return updated

    script = Path(__file__).resolve().with_name("predict_videos.py")
    if not script.is_file():
        raise FileNotFoundError(
            "Place predict_sequence.py beside predict_videos.py."
        )

    print("Selected sequence: " + " -> ".join(labels))

    old_argv = sys.argv[:]
    pipeline.protocol = selected_protocol

    try:
        sys.argv = [str(script), *prediction_args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        pipeline.protocol = original_protocol
        pipeline.analyse_inputs = original_analyse_inputs
        sys.argv = old_argv


if __name__ == "__main__":
    main()
