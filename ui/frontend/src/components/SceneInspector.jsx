import { useState, useRef, useEffect } from "react";
import { Upload, SlidersHorizontal, ChevronLeft, X } from "lucide-react";

function useMediaQuery(query) {
	const [matches, setMatches] = useState(() =>
		typeof window !== "undefined" ? window.matchMedia(query).matches : true
	);
	useEffect(() => {
		const mql = window.matchMedia(query);
		const onChange = e => setMatches(e.matches);
		mql.addEventListener("change", onChange);
		return () => mql.removeEventListener("change", onChange);
	}, [query]);
	return matches;
}

function SceneFields({
	imageType,
	setImageType,
	spatialResolution,
	setSpatialResolution,
	isSimulating,
	setIsSimulating,
	image,
	onUpload,
}) {
	const fileInputRef = useRef(null);

	const handleFileSelect = e => {
		if (e.target.files && e.target.files[0]) {
			onUpload(e.target.files[0]);
			e.target.value = "";
		}
	};

	return (
		<div className="space-y-4 p-3">
			{/* Upload */}
			<div className="flex items-center gap-3">
				<input
					type="file"
					ref={fileInputRef}
					className="hidden"
					accept="image/png, image/jpeg"
					onChange={handleFileSelect}
				/>
				{image ? (
					<img
						src={image}
						alt=""
						className="h-12 w-12 shrink-0 rounded-[10px] border border-line object-cover"
					/>
				) : (
					<div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-[10px] border border-dashed border-line text-muted">
						<Upload className="h-4 w-4" />
					</div>
				)}
				<button
					type="button"
					onClick={() => fileInputRef.current?.click()}
					className="flex flex-1 items-center justify-center gap-2 rounded-full bg-accent-fill px-3 py-2 text-xs font-semibold text-on-accent transition-all duration-150 hover:brightness-110 active:scale-[0.98]"
				>
					<Upload className="h-3.5 w-3.5" />
					{image ? "Replace image" : "Upload EO image"}
				</button>
			</div>

			{/* Type */}
			<div className="flex items-center justify-between gap-3">
				<label htmlFor="scene-image-type" className="text-xs text-muted">
					Type
				</label>
				<select
					id="scene-image-type"
					value={imageType}
					onChange={e => setImageType(e.target.value)}
					className="cursor-pointer rounded-full border border-line bg-surface-solid px-3 py-1.5 text-xs font-medium text-ink transition-colors hover:border-accent/50 focus:border-accent focus:outline-none"
				>
					<option value="RGB">RGB</option>
					<option value="SAR">SAR</option>
					<option value="FCC">FCC</option>
				</select>
			</div>

			{/* Resolution */}
			<div className="flex items-center justify-between gap-3">
				<label htmlFor="scene-resolution" className="text-xs text-muted">
					Resolution (m)
				</label>
				<input
					id="scene-resolution"
					type="number"
					value={spatialResolution}
					onChange={e => setSpatialResolution(parseFloat(e.target.value) || 1.0)}
					min="0.1"
					max="100"
					step="0.1"
					className="w-20 rounded-full border border-line bg-surface-solid px-2 py-1.5 text-center font-mono text-xs font-medium text-ink transition-colors focus:border-accent focus:outline-none"
				/>
			</div>

			{/* Simulation */}
			<div className="flex items-center justify-between gap-3">
				<span className="text-xs text-muted">Simulation</span>
				<button
					type="button"
					role="switch"
					aria-checked={isSimulating}
					onClick={() => setIsSimulating(!isSimulating)}
					className={`relative h-6 w-11 rounded-full border transition-colors duration-200 ${
						isSimulating ? "border-accent-fill bg-accent-fill" : "border-line bg-surface-solid"
					}`}
				>
					<span
						className={`absolute top-0.5 h-4.5 w-4.5 rounded-full bg-white shadow transition-all duration-200 ${
							isSimulating ? "left-[22px]" : "left-0.5"
						}`}
					/>
				</button>
			</div>
		</div>
	);
}

export default function SceneInspector({
	imageType,
	setImageType,
	spatialResolution,
	setSpatialResolution,
	isSimulating,
	setIsSimulating,
	image,
	onUpload,
}) {
	const isDesktop = useMediaQuery("(min-width: 1024px)");
	const [collapsed, setCollapsed] = useState(false);
	const [sheetOpen, setSheetOpen] = useState(false);

	// Desktop: floating panel (collapsible)
	if (isDesktop) {
		if (collapsed) {
			return (
				<button
					type="button"
					onClick={() => setCollapsed(false)}
					title="Scene inspector"
					aria-label="Expand scene inspector"
					className="ui-layer fixed top-1/2 left-3 z-30 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-line bg-surface text-muted shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] transition-all duration-150 hover:text-ink active:scale-95"
				>
					<SlidersHorizontal className="h-4.5 w-4.5" />
				</button>
			);
		}

		return (
			<aside
				aria-label="Scene inspector"
				className="ui-layer fixed top-[76px] left-3 z-30 w-[264px] overflow-y-auto rounded-[20px] border border-line bg-surface shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] no-scrollbar"
				style={{ maxHeight: "calc(100vh - 340px)" }}
			>
				<header className="flex items-center justify-between border-b border-line px-3 py-2.5">
					<h2 className="text-[13px] font-semibold text-ink">Scene</h2>
					<button
						type="button"
						onClick={() => setCollapsed(true)}
						title="Collapse"
						aria-label="Collapse scene inspector"
						className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition-colors hover:bg-line/40 hover:text-ink"
					>
						<ChevronLeft className="h-4 w-4" />
					</button>
				</header>

				<SceneFields
					imageType={imageType}
					setImageType={setImageType}
					spatialResolution={spatialResolution}
					setSpatialResolution={setSpatialResolution}
					isSimulating={isSimulating}
					setIsSimulating={setIsSimulating}
					image={image}
					onUpload={onUpload}
				/>
			</aside>
		);
	}

	// Narrow: trigger button + bottom sheet
	return (
		<>
			<button
				type="button"
				onClick={() => setSheetOpen(true)}
				title="Scene inspector"
				aria-label="Open scene inspector"
				className="ui-layer fixed top-[64px] left-3 z-30 flex h-10 items-center gap-2 rounded-full border border-line bg-surface px-4 text-xs font-semibold text-ink shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] transition-all duration-150 active:scale-95"
			>
				<SlidersHorizontal className="h-4 w-4 text-accent-text" />
				Scene
			</button>

			{sheetOpen && (
				<>
					<div
						className="fixed inset-0 z-40 bg-backdrop/40 animate-fade-up"
						onClick={() => setSheetOpen(false)}
						aria-hidden="true"
					/>
					<aside
						role="dialog"
						aria-label="Scene inspector"
						className="fixed inset-x-0 bottom-0 z-50 max-h-[70vh] overflow-y-auto rounded-t-[20px] border-t border-line bg-surface shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] animate-expand-in"
					>
						<div className="flex items-center justify-between border-b border-line px-3 py-2.5">
							<h2 className="text-[13px] font-semibold text-ink">Scene</h2>
							<button
								type="button"
								onClick={() => setSheetOpen(false)}
								aria-label="Close scene inspector"
								className="flex h-8 w-8 items-center justify-center rounded-full text-muted transition-colors hover:bg-line/40 hover:text-ink"
							>
								<X className="h-4 w-4" />
							</button>
						</div>
						<SceneFields
							imageType={imageType}
							setImageType={setImageType}
							spatialResolution={spatialResolution}
							setSpatialResolution={setSpatialResolution}
							isSimulating={isSimulating}
							setIsSimulating={setIsSimulating}
							image={image}
							onUpload={onUpload}
						/>
					</aside>
				</>
			)}
		</>
	);
}
