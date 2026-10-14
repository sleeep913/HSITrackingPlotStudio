from pathlib import Path

from hsi_plotter.models.project import ProjectSources
from hsi_plotter.validation.project_sources import validate_project_sources


def test_project_source_preflight_requires_core_directories(tmp_path: Path) -> None:
    report = validate_project_sources(ProjectSources())

    assert report.error_count == 2
    assert {issue.code for issue in report.issues} >= {
        "GROUND_TRUTH_NOT_CONFIGURED",
        "RESULTS_ROOT_NOT_CONFIGURED",
    }


def test_project_source_preflight_summarizes_valid_sources(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground-truth"
    results = tmp_path / "results"
    images = tmp_path / "images"
    export = tmp_path / "exports"
    for directory in (ground_truth, results, images, export):
        directory.mkdir()
    (ground_truth / "vis-basketball.txt").write_text("1,2,30,40\n", encoding="utf-8")
    (results / "vis-basketball.txt").write_text("1,2,30,40\n", encoding="utf-8")

    report = validate_project_sources(
        ProjectSources(
            ground_truth=ground_truth,
            results_root=results,
            image_root=images,
            export_directory=export,
        )
    )

    assert not report.has_errors
    assert any(issue.code == "SOURCE_SUMMARY" for issue in report.issues)


def test_project_source_preflight_rejects_wrong_path_kinds(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("data", encoding="utf-8")
    directory = tmp_path / "not-a-file"
    directory.mkdir()

    report = validate_project_sources(
        ProjectSources(
            ground_truth=file_path,
            results_root=file_path,
            attribute_file=directory,
        )
    )

    assert report.has_errors
    assert {issue.code for issue in report.issues} >= {
        "GROUND_TRUTH_DIRECTORY_INVALID",
        "RESULTS_DIRECTORY_INVALID",
        "ATTRIBUTE_FILE_INVALID",
    }


def test_project_source_preflight_accepts_csv_sources(tmp_path: Path) -> None:
    ground_truth = tmp_path / "groundtruth.csv"
    results = tmp_path / "method.csv"
    ground_truth.write_text(
        "dataset,modality,sequence,frame,x,y,width,height\n"
        "HOTC2023,vis,ball,1,1,2,30,40\n",
        encoding="utf-8",
    )
    results.write_text(
        "ID,x,y,width,height\nvis-ball_1,1,2,30,40\n", encoding="utf-8"
    )

    report = validate_project_sources(
        ProjectSources(ground_truth=ground_truth, results_root=results)
    )

    assert not report.has_errors
    assert any(issue.code == "SOURCE_SUMMARY" for issue in report.issues)
