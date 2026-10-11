import { useState, useRef, useEffect } from "react";
import { Upload, Satellite } from "lucide-react";

/**
 * Stage = the full-viewport image layer.
 * The image wrapper + SVG overlay keep the exact coordinate relationship of the
 * original implementation: the overlay is centred on the same flex box that
 * centres the image and is sized from the image's measured client box.
 */
const Stage = ({ image, bboxes = [], onFileDrop }) => {
	const [isDragging, setIsDragging] = useState(false);
	const [imgDims, setImgDims] = useState({ width: 0, height: 0 });
	const imgRef = useRef(null);
	const fileInputRef = useRef(null);
	const onFileDropRef = useRef(onFileDrop);

	useEffect(() => {
		onFileDropRef.current = onFileDrop;
	});

	// Drop anywhere on the window (files only).
	useEffect(() => {
		const hasFiles = e =>
			Array.from((e.dataTransfer && e.dataTransfer.types) || []).includes("Files");

		const onDragEnter = e => {
			if (!hasFiles(e)) return;
			e.preventDefault();
			setIsDragging(true);
		};
		const onDragOver = e => {
			if (!hasFiles(e)) return;
			e.preventDefault();
		};
		const onDragLeave = e => {
			if (!hasFiles(e)) return;
			const leftWindow =
				!e.relatedTarget ||
				e.clientX <= 0 ||
				e.clientY <= 0 ||
				e.clientX >= window.innerWidth ||
				e.clientY >= window.innerHeight;
			if (leftWindow) setIsDragging(false);
		};
		const onDrop = e => {
			setIsDragging(false);
			if (e.defaultPrevented) return;
			const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
			if (!file) return;
			e.preventDefault();
			onFileDropRef.current && onFileDropRef.current(file);
		};
		const onDragEnd = () => setIsDragging(false);

		window.addEventListener("dragenter", onDragEnter);
		window.addEventListener("dragover", onDragOver);
		window.addEventListener("dragleave", onDragLeave);
		window.addEventListener("drop", onDrop);
		window.addEventListener("dragend", onDragEnd);
		return () => {
			window.removeEventListener("dragenter", onDragEnter);
			window.removeEventListener("dragover", onDragOver);
			window.removeEventListener("dragleave", onDragLeave);
			window.removeEventListener("drop", onDrop);
			window.removeEventListener("dragend", onDragEnd);
		};
	}, []);

	// Update dimensions when image loads or window resizes
	useEffect(() => {
		if (!imgRef.current) return;

		const updateDims = () => {
			if (imgRef.current) {
				setImgDims({
					width: imgRef.current.clientWidth,
					height: imgRef.current.clientHeight,
				});
			}
		};

		// Initial check
		updateDims();

		// Observer for precise resizing
		const resizeObserver = new ResizeObserver(() => {
			updateDims();
		});

		resizeObserver.observe(imgRef.current);

		return () => resizeObserver.disconnect();
	}, [image]);

	// Helper to convert bbox to polygon points
	const getPolygonPoints = bbox => {
		const coords = bbox.obbox || bbox;

		// Case 1: 8-point oriented box [x1, y1, x2, y2, x3, y3, x4, y4] (Normalized 0-1)
		if (Array.isArray(coords) && coords.length >= 8) {
			// Scale normalized 0-1 to SVG viewBox 0-100
			return coords
				.slice(0, 8)
				.map(c => c * 100)
				.reduce((acc, val, i, arr) => {
					if (i % 2 === 0) acc.push(`${val},${arr[i + 1]}`);
					return acc;
				}, [])
				.join(" ");
		}

		// Case 2: Fallback for 4-point rect [x, y, w, h] (Percentages 0-100)
		if (Array.isArray(coords) && coords.length === 4) {
			const [x, y, w, h] = coords;
			return `${x},${y} ${x + w},${y} ${x + w},${y + h} ${x},${y + h}`;
		}
		return "";
	};

	const handleFileSelect = e => {
		if (e.target.files && e.target.files[0]) {
			onFileDrop(e.target.files[0]);
			e.target.value = "";
		}
	};

	return (
		<div className="absolute inset-0 z-0 overflow-hidden bg-backdrop">
			<input
				type="file"
				ref={fileInputRef}
				className="hidden"
				accept="image/png, image/jpeg"
				onChange={handleFileSelect}
			/>

			{image ? (
				<div className="absolute inset-0 flex items-center justify-center">
					{/* Image Element */}
					<img
						ref={imgRef}
						src={image}
						alt="Satellite View"
						className="max-h-full max-w-full object-contain"
						onLoad={() => {
							if (imgRef.current) {
								setImgDims({
									width: imgRef.current.clientWidth,
									height: imgRef.current.clientHeight,
								});
							}
						}}
					/>

					{/* Overlay for Bounding Boxes (SVG matches image size exactly) */}
					<svg
						className="pointer-events-none absolute"
						style={{
							width: imgDims.width,
							height: imgDims.height,
							// Center the absolute SVG over the image (which is centered by flex)
							top: "50%",
							left: "50%",
							transform: "translate(-50%, -50%)",
						}}
						viewBox="0 0 100 100"
						preserveAspectRatio="none"
					>
						{bboxes.map((bbox, idx) => {
							const points = getPolygonPoints(bbox);
							if (!points) return null;

							const nums = points.split(/[\s,]+/).map(Number);
							const cx =
								nums.filter((_, i) => i % 2 === 0).reduce((a, b) => a + b, 0) /
								(nums.length / 2);
							const cy =
								nums.filter((_, i) => i % 2 === 1).reduce((a, b) => a + b, 0) /
								(nums.length / 2);

							return (
								<g key={idx}>
									<polygon points={points} className="bbox-polygon" />
									<text
										x={cx}
										y={cy}
										dy="0.5"
										textAnchor="middle"
										className="bbox-label text-[3px]"
										style={{ textShadow: "0px 0px 1px rgba(255,255,255,0.9)" }}
									>
										{idx + 1}
									</text>
								</g>
							);
						})}
					</svg>
				</div>
			) : (
				<div
					onClick={() => fileInputRef.current?.click()}
					className="absolute inset-0 flex cursor-pointer flex-col items-center justify-center px-6 text-center"
				>
					<div className="mb-6 flex h-28 w-28 items-center justify-center rounded-[28px] border-2 border-dashed border-accent/50 bg-accent-soft shadow-[var(--float-shadow)]">
						<Satellite className="h-12 w-12 text-accent-text" strokeWidth={1.5} />
					</div>
					<h1 className="font-display text-4xl leading-tight font-normal text-ink">
						Drag &amp; drop image
					</h1>
					<p className="mt-3 text-sm text-ink/70">
						Earth observation scenes — PNG or JPG, up to the size your mission needs.
					</p>
					<p className="mt-1 font-mono text-xs text-ink/60">PNG · JPG supported</p>
					<button
						type="button"
						onClick={e => {
							e.stopPropagation();
							fileInputRef.current?.click();
						}}
						className="mt-6 flex items-center gap-2 rounded-full bg-accent-fill px-6 py-2.5 text-sm font-semibold text-on-accent shadow-[var(--float-shadow)] transition-all duration-150 hover:brightness-110 active:scale-[0.98]"
					>
						<Upload className="h-4 w-4" /> Browse files
					</button>
				</div>
			)}

			{/* Full-window drag overlay */}
			{isDragging && (
				<div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center bg-backdrop/70 backdrop-blur-[2px]">
					<div className="flex flex-col items-center rounded-[24px] border-2 border-dashed border-accent bg-surface px-14 py-10 shadow-[var(--float-shadow)] animate-rise-in">
						<Upload className="mb-3 h-10 w-10 text-accent-text" />
						<p className="font-display text-2xl text-ink">Drop image here</p>
						<p className="mt-1 text-sm text-ink/70">Release to load it onto the stage</p>
					</div>
				</div>
			)}
		</div>
	);
};

export default Stage;
