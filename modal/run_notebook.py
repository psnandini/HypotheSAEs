"""Run one of the Colab notebooks on a Modal GPU instead of Colab.

    modal run modal/run_notebook.py --notebook notebooks/followups_drug_reviews.ipynb

The notebook is executed top to bottom in a GPU container. Its Colab-only import
line is replaced by small stand-ins: `userdata.get("MODAL_VLLM_KEY")` reads the
vLLM API key from the same Modal secret the server uses, and `files.download` does
nothing (result files are copied back instead). The executed notebook is written to
`<name>_executed.ipynb` next to the original, and every file the notebook saved in
/content/results is copied to results/.
"""
import pathlib

import modal

GPU = ["T4", "L4"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "unzip", "git")
    .uv_pip_install("git+https://github.com/psnandini/HypotheSAEs.git", "nbconvert", "ipykernel", "matplotlib")
)
app = modal.App("hypothesaes-notebook", image=image)

# Replaces the notebook's `from google.colab import ...` line.
COLAB_IMPORT = "from google.colab import userdata, files"
COLAB_STUB = (
    "import types\n"
    "userdata = types.SimpleNamespace(get=lambda name: os.environ['VLLM_API_KEY'])\n"
    "files = types.SimpleNamespace(download=lambda path: print(f'(saved {path})'))"
)


@app.function(gpu=GPU, timeout=3 * 60 * 60, secrets=[modal.Secret.from_name("hypothesaes-vllm-key")])
def execute(notebook_json: str) -> dict:
    import os

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    os.makedirs("/content", exist_ok=True)

    nb = nbformat.reads(notebook_json, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and COLAB_IMPORT in cell.source:
            cell.source = cell.source.replace(COLAB_IMPORT, COLAB_STUB)
    error = None
    try:
        ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(nb, {"metadata": {"path": "/content"}})
    except Exception as e:  # keep the partly executed notebook so the failing cell can be inspected
        error = str(e)[-3000:]

    results_dir = pathlib.Path("/content/results")
    files = {p.name: p.read_bytes() for p in results_dir.glob("*")} if results_dir.exists() else {}
    return {"notebook": nbformat.writes(nb), "files": files, "error": error}


@app.local_entrypoint()
def main(notebook: str):
    path = pathlib.Path(notebook)
    out = execute.remote(path.read_text(encoding="utf-8"))
    executed = path.with_name(path.stem + "_executed.ipynb")
    executed.write_text(out["notebook"], encoding="utf-8")
    results = path.parent.parent / "results"
    results.mkdir(exist_ok=True)
    for name, data in out["files"].items():
        (results / name).write_bytes(data)
    print(f"wrote {executed} and {len(out['files'])} result files to {results}")
    if out["error"]:
        print("NOTEBOOK ERROR:\n" + out["error"])
