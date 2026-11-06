"""Evaluation runner of the thesis, split into components:

- ``cli``          command line (``python evaluate.py``)
- ``runner``       run, dataset and system orchestration, phases, concurrency limits
- ``metrics``      registry of the metrics and their compute functions
- ``llm_judges``   FS, GS and Coherence: prompts, response parsers, scores
- ``inputs``       loading predictions, gold and MINE graphs; triple statistics; embeddings
- ``results``      result files: incremental atomic saving, compact form
- ``summary``      run summary with the status of every metric
- ``concurrency``  bounded gathering of LLM requests with cancellation on failure
- ``gpu``          GPU selection and memory release
"""
