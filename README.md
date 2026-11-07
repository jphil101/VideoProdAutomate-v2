# VideoProdAutomate

Two pipelines live here:

* **`simreel/` (new)**: describe construction events and supply a narration script, and get a Blender-rendered video with crane lifts, rigid-body settling, physics QA, word-synced captions (burned in and as a soft track), and optional stock cut-ins (Pexels, Pixabay, Envato, or a local folder). See [`simreel/README.md`](simreel/README.md).

  ```bash
  python -m simreel make --job examples/modular_frame/job.yaml
  ```

* **`main_workflow.py` (legacy)**: a stock-footage montage with TTS and Whisper captions. See `Video Production Workflow Flowchart v2.md`. Its `EnvatoDownloader.py` is reused by simreel's `envato` stock provider.
