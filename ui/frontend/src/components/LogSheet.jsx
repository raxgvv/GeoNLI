import { useRef, useEffect } from "react";
import { X, Pin, MapPin, MessageSquare } from "lucide-react";

export default function LogSheet({
	open,
	onClose,
	pinned,
	onTogglePinned,
	messages,
	isLoading,
}) {
	const listRef = useRef(null);

	useEffect(() => {
		if (open && listRef.current) {
			listRef.current.scrollTop = listRef.current.scrollHeight;
		}
	}, [open, messages, isLoading]);

	return (
		<aside
			aria-label="Conversation log"
			aria-hidden={!open}
			inert={!open}
			className={`ui-layer fixed top-[68px] right-3 z-30 flex w-[min(400px,calc(100vw-24px))] flex-col overflow-hidden rounded-[20px] border border-line bg-surface shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] transition-all duration-200 ${
				open
					? "pointer-events-auto translate-x-0 opacity-100"
					: "pointer-events-none translate-x-6 opacity-0"
			}`}
			style={{ maxHeight: "calc(100vh - 180px)" }}
		>
			<header className="flex shrink-0 items-center justify-between gap-2 border-b border-line px-4 py-3">
				<h2 className="text-[13px] font-semibold text-ink">Conversation log</h2>
				<div className="flex items-center gap-1">
					<button
						type="button"
						onClick={onTogglePinned}
						aria-pressed={pinned}
						title={pinned ? "Unpin (auto-close resumes)" : "Pin (keep open)"}
						className={`flex h-8 w-8 items-center justify-center rounded-full transition-colors duration-150 ${
							pinned ? "bg-info-soft text-info" : "text-muted hover:bg-line/40 hover:text-ink"
						}`}
					>
						<Pin className={`h-4 w-4 ${pinned ? "rotate-45" : ""}`} />
					</button>
					<button
						type="button"
						onClick={onClose}
						title="Dismiss"
						aria-label="Dismiss conversation log"
						className="flex h-8 w-8 items-center justify-center rounded-full text-muted transition-colors duration-150 hover:bg-line/40 hover:text-ink"
					>
						<X className="h-4 w-4" />
					</button>
				</div>
			</header>

			<div ref={listRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
				{messages.length === 0 && (
					<div className="flex flex-col items-center py-8 text-center">
						<div className="mb-3 flex h-12 w-12 items-center justify-center rounded-[14px] border border-line bg-surface-solid text-muted">
							<MessageSquare className="h-6 w-6" />
						</div>
						<p className="text-sm text-muted">No messages yet.</p>
					</div>
				)}

				{messages.map((msg, idx) => (
					(() => {
						const parsedTimestamp = new Date(msg.timestamp);
						const safeTimestamp = Number.isNaN(parsedTimestamp.getTime()) ? new Date() : parsedTimestamp;
						return (
					<div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
						<div
							className={`max-w-[88%] rounded-2xl p-3 text-sm leading-relaxed ${
								msg.role === "user"
									? "rounded-br-[6px] bg-accent-fill text-on-accent"
									: "rounded-bl-[6px] border border-line bg-surface-solid text-ink"
							} ${msg.isError ? "border-danger/40 bg-danger-soft text-danger" : ""}`}
						>
							<p>{msg.content}</p>
							{msg.bboxes && msg.bboxes.length > 0 && (
								<div className="mt-2 flex items-center gap-2 border-t border-line pt-2 text-xs text-accent-text">
									<MapPin className="h-3 w-3" />
									<span>Highlighted {msg.bboxes.length} region(s)</span>
								</div>
							)}
							{msg.role === "user" && msg.task && (
								<div className="mt-1.5 flex justify-end gap-1">
									<span className="rounded-full bg-black/20 px-2 py-0.5 text-[10px] font-medium opacity-80">
										{msg.task}
									</span>
									{msg.subtask && (
										<span className="rounded-full bg-black/20 px-2 py-0.5 text-[10px] font-medium opacity-80">
											{msg.subtask}
										</span>
									)}
								</div>
							)}
							<div
								className={`mt-1 text-right font-mono text-[10px] ${
									msg.role === "user" ? "opacity-70" : "text-muted"
								}`}
							>
								{safeTimestamp.toLocaleTimeString([], {
									hour: "2-digit",
									minute: "2-digit",
								})}
							</div>
						</div>
					</div>
						);
					})()
				))}

				{isLoading && (
					<div className="flex justify-start">
						<div className="flex items-center gap-2 rounded-2xl rounded-bl-[6px] border border-line bg-surface-solid p-3">
							<div className="h-2 w-2 animate-bounce rounded-full bg-accent [animation-delay:0ms]" />
							<div className="h-2 w-2 animate-bounce rounded-full bg-accent [animation-delay:150ms]" />
							<div className="h-2 w-2 animate-bounce rounded-full bg-accent [animation-delay:300ms]" />
						</div>
					</div>
				)}
			</div>
		</aside>
	);
}
