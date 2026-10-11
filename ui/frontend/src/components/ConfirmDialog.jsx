import { useEffect, useRef } from "react";
import { AlertTriangle } from "lucide-react";

export default function ConfirmDialog({
	isOpen,
	title,
	message,
	confirmLabel = "Confirm",
	cancelLabel = "Cancel",
	onConfirm,
	onCancel,
}) {
	const cancelRef = useRef(null);

	useEffect(() => {
		if (isOpen && cancelRef.current) {
			cancelRef.current.focus();
		}
	}, [isOpen]);

	useEffect(() => {
		if (!isOpen) return;
		const onKeyDown = e => {
			if (e.key === "Escape") onCancel();
		};
		window.addEventListener("keydown", onKeyDown);
		return () => window.removeEventListener("keydown", onKeyDown);
	}, [isOpen, onCancel]);

	if (!isOpen) return null;

	return (
		<div
			className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm"
			onClick={onCancel}
			role="dialog"
			aria-modal="true"
			aria-label={title}
		>
			<div
				className="animate-rise-in w-full max-w-sm rounded-[var(--radius-xl)] border border-line bg-surface-solid p-5 shadow-[var(--float-shadow)]"
				onClick={e => e.stopPropagation()}
			>
				<div className="flex items-start gap-3">
					<div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-danger-soft text-danger">
						<AlertTriangle className="h-4.5 w-4.5" />
					</div>
					<div className="min-w-0 flex-1">
						<h2 className="font-display text-base font-semibold text-ink">{title}</h2>
						<p className="mt-1 text-sm leading-relaxed text-muted">{message}</p>
					</div>
				</div>

				<div className="mt-5 flex items-center justify-end gap-2">
					<button
						ref={cancelRef}
						type="button"
						onClick={onCancel}
						className="flex h-9 items-center rounded-full border border-line bg-surface-solid px-4 text-sm font-medium text-muted transition-all duration-150 hover:text-ink active:scale-95"
					>
						{cancelLabel}
					</button>
					<button
						type="button"
						onClick={onConfirm}
						className="flex h-9 items-center rounded-full bg-danger px-4 text-sm font-semibold text-white transition-all duration-150 hover:brightness-110 active:scale-95"
					>
						{confirmLabel}
					</button>
				</div>
			</div>
		</div>
	);
}
