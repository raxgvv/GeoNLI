import { useState } from "react";
import { Sparkles, ChevronDown, ChevronUp, Copy, Check } from "lucide-react";

export default function CaptionCard({
	caption,
	image,
	isGeneratingCaption,
	onGenerateCaption,
	onCaptionChange,
}) {
	const [isOpen, setIsOpen] = useState(true);
	const [copied, setCopied] = useState(false);

	const handleCopy = async () => {
		if (!caption) return;
		try {
			await navigator.clipboard.writeText(caption);
			setCopied(true);
			setTimeout(() => setCopied(false), 1500);
		} catch {
			/* clipboard unavailable — keep existing behaviour silently */
		}
	};

	// Empty: collapse to a chip
	if (!caption && !isGeneratingCaption) {
		return (
			<button
				type="button"
				onClick={onGenerateCaption}
				disabled={!image}
				title={!image ? "Upload an image first" : "Generate caption"}
				className="ui-layer fixed bottom-[96px] left-3 z-30 flex items-center gap-2 rounded-full border border-line bg-surface px-4 py-2.5 text-xs font-semibold text-ink shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] transition-all duration-150 hover:border-accent/50 hover:text-accent-text active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 max-[1023px]:left-3 max-[1023px]:right-3 max-[1023px]:justify-center"
			>
				<Sparkles className="h-4 w-4 text-accent-text" />
				Scene caption
			</button>
		);
	}

	return (
		<section
			aria-label="Scene caption"
			className="ui-layer fixed bottom-[96px] left-3 z-30 w-[min(360px,calc(100vw-24px))] overflow-hidden rounded-[20px] border border-line bg-surface shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] animate-rise-in max-[1023px]:right-3 max-[1023px]:w-auto"
		>
			<header className="flex items-center justify-between gap-2 border-b border-line px-3 py-2">
				<button
					type="button"
					onClick={() => setIsOpen(open => !open)}
					aria-expanded={isOpen}
					className="flex items-center gap-2 text-xs font-semibold text-ink transition-colors hover:text-accent-text"
				>
					{isOpen ? (
						<ChevronDown className="h-3.5 w-3.5 text-muted" />
					) : (
						<ChevronUp className="h-3.5 w-3.5 text-muted" />
					)}
					<Sparkles className="h-3.5 w-3.5 text-accent-text" />
					Scene caption
				</button>

				<div className="flex items-center gap-1.5">
					<button
						type="button"
						onClick={handleCopy}
						disabled={!caption}
						title={caption ? "Copy caption" : "No caption to copy"}
						className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-[11px] font-medium text-muted transition-colors duration-150 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
					>
						{copied ? (
							<Check className="h-3.5 w-3.5 text-success" />
						) : (
							<Copy className="h-3.5 w-3.5" />
						)}
						{copied ? "Copied" : "Copy"}
					</button>
				</div>
			</header>

			{isOpen && (
				<div className="flex items-stretch gap-3 p-3">
					<button
						type="button"
						onClick={onGenerateCaption}
						disabled={!image || isGeneratingCaption}
						title={!image ? "Upload an image first" : "Generate caption"}
						className="group flex w-24 shrink-0 flex-col items-center justify-center gap-1 rounded-[14px] border border-accent/40 bg-accent-soft px-2 py-3 transition-all duration-150 hover:border-accent active:scale-[0.98] disabled:cursor-not-allowed disabled:border-line disabled:bg-line/30 disabled:opacity-60"
					>
						{isGeneratingCaption ? (
							<div className="h-5 w-5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
						) : (
							<Sparkles className="h-5 w-5 text-accent-text transition-transform duration-150 group-hover:scale-110" />
						)}
						<span className="text-center text-[11px] font-semibold leading-tight text-accent-text">
							Generate
							<br />
							caption
						</span>
					</button>

					<div className="relative min-w-0 flex-1">
						<textarea
							value={caption || ""}
							onChange={e => onCaptionChange?.(e.target.value)}
							placeholder={
								image
									? "Click button to analyze scene contents or give prompt for generation..."
									: "Upload image to enable captioning."
							}
							disabled={!image}
							className="h-[110px] w-full resize-none rounded-[14px] border border-line bg-surface-solid p-3 text-sm leading-relaxed text-ink placeholder:text-xs placeholder:italic placeholder:text-muted focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/40 transition-all disabled:cursor-not-allowed"
						/>
						{isGeneratingCaption && (
							<div className="shimmer pointer-events-none absolute inset-0 rounded-[14px]" />
						)}
					</div>
				</div>
			)}
		</section>
	);
}
