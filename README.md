# LLM Research Judge

LLM-assisted scientific abstract and manuscript evaluation.

This repository contains the code used for abstract evaluation in the
[18th Professor Alborzi International Congress of Clinical Microbiology (PAICCM2025)](https://congress.alborzicmrc.ir/index.php).
It includes utilities for parsing abstracts, screening submissions, running structured LLM evaluation, and exporting results.

## Main Workflow

The notebook [`evaluator.ipynb`](evaluator.ipynb) is the recommended entry point. The primary evaluation path used there is:

```python
from Code.generator import prof_alborzi_congress_eval_async
from Code.enhanced_schemas import abstract_criteria_definer

parser = abstract_criteria_definer()

results = await prof_alborzi_congress_eval_async(
    data_path="not_narrative.json",
    prompt=prompt_abs,
    parser=parser,
    model="gpt-4.1-mini",
    concurrency_limit=15,
)
```

Set `OPENAI_API_KEY` in a `.env` file or in your shell before running model calls.

## Optional Full-Schema Evaluation

The split/asynchronous workflow is the notebook-facing path. A full, unseparated schema evaluation is still available for direct use:

```python
from Code.complete_workflow import run_complete_evaluation

excel_path, json_path = run_complete_evaluation(
    parsed_abstracts_file="parsed_abstracts.json",
    output_dir="evaluation_results",
    model="gpt-4.1-mini",
)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On non-Windows systems, `textract` may require additional system packages. If you only work with Excel, JSON, PDF, or DOCX inputs, the parser utilities can usually be adapted to use `pandas`, `PyMuPDF`, `python-docx`, or `docx2txt` directly.

## Repository Layout

- `evaluator.ipynb`: notebook workflow and examples
- `Code/generator.py`: OpenAI/OpenRouter request helpers and async congress evaluator
- `Code/enhanced_schemas.py`: main structured evaluation schema
- `Code/eval_complete.py`: optional full-schema evaluation workflow
- `Code/eval_split.py`: optional split-schema evaluation workflow
- `Code/robust_abstract_parser.py`: Excel/DOCX/PDF abstract parsing helpers
- `Code/screen.py`: similarity and screening utilities
- `Code/utils.py`: document loading and export helpers
