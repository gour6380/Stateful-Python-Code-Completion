"""Notebook ordering, configuration selection, and import-safety contracts."""

import ast
import json
from pathlib import Path

from src.config import DataConfig, ExperimentConfig, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_has_one_action_per_cell():
    notebook = json.loads((ROOT / "notebooks" / "stateful_code_completion.ipynb").read_text())
    previous = None
    actions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert previous == "markdown", "Every action needs its explanation"
            # Outputs are forbidden at handoff (checked statically), but the
            # owner may have autosaved earlier cells before invoking this suite.
            action = cell["metadata"]["action"]
            assert action not in actions
            actions.append(action)
            tree = ast.parse("".join(cell["source"]))
            if action not in {"imports", "pipeline_imports"}:
                assert len(tree.body) == 1, f"Cell {action} has multiple top-level operations"
        previous = cell["cell_type"]
    assert len(actions) == 68
    assert notebook["metadata"]["kernelspec"]["name"] == "stateful-code-completion-py313"


def test_no_import_time_pipeline_invocations():
    forbidden = {
        "create_run",
        "acquire_data",
        "prepare_data",
        "train_model",
        "start_service",
        "evaluate_model",
        "evaluate_lines",
        "evaluate_streaming",
        "benchmark_engine",
        "benchmark_http",
        "generate_report",
        "parse_args",
    }
    for path in (ROOT / "src").glob("*.py"):
        tree = ast.parse(path.read_text())
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for node in ast.walk(statement):
                if isinstance(node, ast.Call):
                    name = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else getattr(node.func, "attr", "")
                    )
                    assert name not in forbidden, (path.name, name)


def test_notebook_bootstraps_effective_environment_before_torch_imports():
    notebook = json.loads((ROOT / "notebooks" / "stateful_code_completion.ipynb").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    actions = [cell["metadata"]["action"] for cell in code]
    required_order = [
        "training_config",
        "run_selection",
        "effective_configuration",
        "mps_environment",
        "pipeline_imports",
        "execution_settings",
        "run",
        "model",
    ]
    assert [actions.index(action) for action in required_order] == sorted(
        actions.index(action) for action in required_order
    )
    for cell in code[: actions.index("mps_environment")]:
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Import):
                assert all(name.name in {"json", "sys"} for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module in {"pathlib", "src.config", "src.bootstrap"}
    environment_cell = code[actions.index("mps_environment")]
    call = ast.parse("".join(environment_cell["source"])).body[0].value
    assert ast.unparse(call) == "configure_environment(effective_config.training.prefer_metal)"
    execution_cell = code[actions.index("execution_settings")]
    call = ast.parse("".join(execution_cell["source"])).body[0].value
    assert ast.unparse(call) == "configure_execution(effective_config.training)"


def test_notebook_explicit_pilot_and_saved_configuration_precedence_without_execution(tmp_path):
    notebook = json.loads((ROOT / "notebooks" / "stateful_code_completion.ipynb").read_text())
    cells = {
        cell["metadata"]["action"]: ast.parse("".join(cell["source"]))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    data_call = cells["data_config"].body[0].value
    assert isinstance(data_call, ast.Call) and ast.unparse(data_call.func) == "DataConfig"
    data_options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in data_call.keywords}
    assert data_options == {"training_files": 5000, "validation_files": 1000}
    training_call = cells["training_config"].body[0].value
    assert (
        isinstance(training_call, ast.Call) and ast.unparse(training_call.func) == "TrainingConfig"
    )
    training_options = {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in training_call.keywords
    }
    assert training_options == {"max_epochs": 15}
    assert ast.literal_eval(cells["run_selection"].body[0].value) is None
    # Inspect the actual branch expression without executing any notebook cell.
    expected = ast.parse(
        "experiment_config if RUN_NAME is None else ExperimentConfig.from_dict("
        'json.loads((PROJECT_ROOT / "runs" / RUN_NAME / "config" / "experiment.json").read_text()))',
        mode="eval",
    ).body
    assert ast.dump(cells["effective_configuration"].body[0].value) == ast.dump(expected)
    config = ExperimentConfig(
        data=DataConfig(**data_options), training=TrainingConfig(**training_options)
    )
    chosen = config.training
    assert (chosen.batch_size, chosen.accumulation_steps, chosen.validation_batch_size) == (
        4,
        8,
        16,
    )
    assert (chosen.cpu_threads, chosen.num_workers, chosen.prefer_metal) == (6, 0, True)
    assert (chosen.min_epochs, chosen.max_epochs, chosen.patience) == (5, 15, 3)
    saved = config.to_dict()
    saved["data"].update(training_files=20000, validation_files=None)
    saved["training"].update(
        batch_size=2,
        accumulation_steps=16,
        validation_batch_size=2,
        cpu_threads=1,
        prefer_metal=False,
        max_epochs=20,
    )
    path = tmp_path / "runs" / "run_003" / "config" / "experiment.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(saved))
    restored = ExperimentConfig.from_dict(json.loads(path.read_text()))
    assert restored.to_dict() == saved
    # Legacy saved runs omitted this field and must retain their uncapped validation pool.
    legacy = json.loads(path.read_text())
    del legacy["data"]["validation_files"]
    assert ExperimentConfig.from_dict(legacy).data.validation_files is None
