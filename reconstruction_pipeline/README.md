# reconstruction_pipeline

Domain boundary for 2D-to-3D reconstruction workflows.

Scope:
- 2D keypoint extraction
- time alignment / triangulation / 3D reconstruction
- evaluation and reconstruction debug tools

Safe split note:
- Existing code still lives under legacy paths during migration.
- New reconstruction tools should be placed under this domain.

Canonical path:
- reconstruction_pipeline/algorithm_pipeline/

Compatibility alias:
- root algorithm_pipeline -> reconstruction_pipeline/algorithm_pipeline (symlink)
