export default function DebugPanel({ isOpen, onClose, info }) {
	if (!isOpen) return null;

	return (
		<div className="fixed right-3 bottom-4 z-50 max-w-md rounded-[20px] border border-line bg-surface p-4 font-mono text-xs text-ink shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)]">
			<h4 className="mb-2 flex items-center justify-between gap-4 border-b border-line pb-1 font-sans text-[13px] font-semibold text-ink">
				<span>API debugger</span>
				<button
					type="button"
					onClick={onClose}
					aria-label="Close debugger"
					className="font-mono text-muted transition-colors hover:text-danger"
				>
					x
				</button>
			</h4>
			<div className="space-y-1">
				{Object.entries(info).map(([key, value]) => (
					<p key={key} className="flex gap-2">
						<span className="min-w-[60px] font-bold text-muted uppercase">{key}:</span>
						<span className="truncate">
							{typeof value === "object" && value !== null
								? JSON.stringify(value)
								: String(value)}
						</span>
					</p>
				))}
				{info.lastResponse && (
					<pre className="mt-2 max-h-40 overflow-hidden overflow-y-auto rounded-[14px] border border-line bg-surface-solid p-2">
						{JSON.stringify({ lastResponse: info.lastResponse }, null, 2)}
					</pre>
				)}
			</div>
		</div>
	);
}
