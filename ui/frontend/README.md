# GeoNLI Frontend (ISRO Submission)

This React + Vite application is the human-in-the-loop console we are submitting for the ISRO GeoNLI track. It demonstrates natural language interaction with electro-optical (EO) satellite chips for three mission tasks: captioning, visual grounding, and question answering (binary/numeric/semantic).

## System Explanation
- **Task Hub**: Users choose between Grounding and VQA in the floating command bar at the bottom of the stage. VQA exposes binary (yes/no), numeric (counts/measurements), and semantic (descriptive) sub-modes so evaluators can probe different reasoning paths without reloading the page.
- **Captioning Assist**: A dedicated caption generator summarises the active scene and stores it with the session, enabling quick orientation before Q&A.
- **Spatial Feedback**: Grounding responses render bounding boxes over the EO chip using percentage coordinates returned by the backend, making evidence explicit.
- **Debug + Simulation**: A toggle in the Scene inspector switches between the live GPU backend and a deterministic simulator that mirrors expected JSON schemas. The debug panel (settings icon in the top strip) displays the last payload/response pair in case verification of inference behaviour is required during judging.
- **UX Considerations**: Drag-and-drop ingestion anywhere on the stage, adaptive prompts, and light/dark theming (light by default) minimise friction when running marathon evaluations.

## Evaluation Instructions
1. **Install Dependencies** (Node 20+/npm 10+ required):
	```bash
	npm install
	```
2. **Configure Backend**: Edit `.env` file and set `VITE_BACKEND_BASE_URL` to the provided GPU endpoint (default placeholder: `<backend-ip>:8000`). The UI invokes `POST /geoNLI/eval` with a `multipart/form-data` payload containing:
	- `image`: uploaded EO chip (PNG/JPEG)
	- `task_type`: `captioning`, `grounding`, or `vqa`
    - `image_type`: `RGB`, `SAR` or `FCC`
	- `vqa_type`: `binary`, `numeric`, or `semantic` (only when `task_type === 'vqa'`)
	- `query`: natural language instruction (empty string for captioning)
	- `spatial_resolution_m`: ground sample distance in metres (stringified number)
3. **Run the Interface**:
	```bash
	npm run dev
	```
	Open the printed URL (default `http://localhost:5173`). For production validation, build and preview:
	```bash
	npm run build
	npm run preview
	```
4. **Workflow**:
	- Upload or drag an EO chip from the Scene inspector or by dropping the file anywhere on the stage.
	- Optionally create additional sessions with the **+** tab in the session strip to compare multiple test images.
	- Trigger **Generate Caption** to obtain the model's scene summary.
	- Select **Grounding** or **VQA** in the command bar (VQA subtype pills expand above it), enter the evaluation prompt, then send.
	- Observe textual answers plus overlays (for Grounding) directly on the imagery.
	- Switch to **Simulation** mode if the backend is unavailable; the UI returns deterministic canned responses while preserving all interaction mechanics.
	- Open the **Debug Panel** (settings icon in the top strip) to inspect the latest API exchange, useful when documenting results or investigating anomalies.

## Reference Scripts

| Command | Purpose |
| ------- | ------- |
| `npm run dev` | Start the evaluation UI with hot reload. |
| `npm run build` | Produce the optimized bundle submitted for scoring. |
| `npm run preview` | Serve the built assets to mimic deployment. |
| `npm run lint` | Sanity-check JSX/React patterns (optional for judging). |

## Notes for Deployment Review
- All bounding boxes are interpreted as normalised coordinates relative to the uploaded image. The backend must follow this format for overlays to align.
- Images remain in-memory via `URL.createObjectURL` for privacy; refreshing the page clears them.
- Theme preference resets on reload to enforce a consistent initial appearance during judging.
- No external analytics or telemetry is embedded; every network call targets the configured backend host only.

