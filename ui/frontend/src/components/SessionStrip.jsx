import { Plus, Image as ImageIcon, Sun, Moon, Cpu, MessagesSquare, X } from "lucide-react";
import StatusBadge from "./StatusBadge";

export default function SessionStrip({
	sessions,
	activeSessionId,
	onSessionChange,
	onCreateSession,
	onTerminateSession,
	activeSession,
	isDarkMode,
	onToggleTheme,
	onToggleDebug,
	apiStatus,
	logOpen,
	onToggleLog,
	messageCount,
}) {
	return (
		<div className="ui-layer fixed top-0 right-0 left-0 z-40 flex h-14 items-center gap-3 border-b border-line bg-surface px-3 shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)]">
			{/* Brand tag */}
			<div className="flex shrink-0 items-center gap-2 pr-1">
				<img src="/isro.svg" alt="ISRO Logo" className="h-7 w-7" />
				<span className="font-display text-lg leading-none font-medium text-ink">
					GeoNLI
				</span>
			</div>

			<span className="hidden h-6 w-px shrink-0 bg-line md:block" />

			{/* Session tabs */}
			<nav
				aria-label="Sessions"
				className="flex min-w-0 flex-1 items-end gap-1 overflow-x-auto pr-2 no-scrollbar"
			>
				{sessions.map(session => {
					const isActive = activeSessionId === session.id;
					return (
						<button
							key={session.id}
							type="button"
							onClick={() => onSessionChange(session.id)}
							aria-current={isActive ? "true" : undefined}
							title={session.name}
							className={`group flex max-w-[190px] shrink-0 items-center gap-2 rounded-t-[14px] rounded-b-[6px] border px-3 py-2 text-sm transition-all duration-150 ${
								isActive
									? "translate-y-px border-accent-fill bg-accent-fill font-semibold text-on-accent shadow-[var(--float-shadow)]"
									: "border-transparent text-muted hover:border-line hover:bg-surface-solid hover:text-ink"
							}`}
						>
							<span
								className={`flex h-5 w-5 shrink-0 items-center justify-center overflow-hidden rounded-[5px] ${
									isActive ? "bg-black/20" : "bg-line/60"
								}`}
							>
								{session.image ? (
									<img src={session.image} alt="" className="h-full w-full object-cover" />
								) : (
									<ImageIcon className="h-3 w-3 opacity-70" />
								)}
							</span>
						<span className="truncate">{session.name}</span>
						<span
							role="button"
							tabIndex={0}
							title={`Terminate ${session.name}`}
							aria-label={`Terminate ${session.name}`}
							onClick={e => {
								e.stopPropagation();
								onTerminateSession(session.id);
							}}
							onKeyDown={e => {
								if (e.key === "Enter" || e.key === " ") {
									e.stopPropagation();
									e.preventDefault();
									onTerminateSession(session.id);
								}
							}}
							className={`ml-1 flex h-5 w-5 shrink-0 items-center justify-center rounded-full transition-all duration-150 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 hover:bg-danger-soft hover:text-danger ${
								isActive ? "opacity-60" : ""
							}`}
						>
							<X className="h-3 w-3" />
						</span>
					</button>
				);
				})}

				<button
					type="button"
					onClick={onCreateSession}
					title="New Session"
					aria-label="New Session"
					className="mb-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-dashed border-line text-muted transition-all duration-150 hover:border-accent hover:text-accent-text active:scale-95"
				>
					<Plus className="h-4 w-4" />
				</button>
			</nav>

			{/* Right controls — pr leaves room for the floating Hide-UI button */}
			<div className="flex shrink-0 items-center gap-2 pr-14">
				<StatusBadge status={apiStatus} />

				<button
					type="button"
					onClick={onToggleLog}
					aria-pressed={logOpen}
					title="Conversation log"
					className={`relative flex h-9 w-9 items-center justify-center rounded-full border transition-all duration-150 active:scale-95 ${
						logOpen
							? "border-info/50 bg-info-soft text-info"
							: "border-line bg-surface-solid text-muted hover:text-ink"
					}`}
				>
					<MessagesSquare className="h-4 w-4" />
					{messageCount > 0 && (
						<span className="absolute -top-1 -right-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-accent-fill px-1 font-mono text-[10px] leading-none text-on-accent">
							{messageCount}
						</span>
					)}
				</button>

				<button
					type="button"
					onClick={onToggleTheme}
					title={isDarkMode ? "Switch to light theme" : "Switch to dark theme"}
					aria-label="Toggle theme"
					className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface-solid text-muted transition-all duration-150 hover:text-ink active:scale-95"
				>
					{isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
				</button>

				<button
					type="button"
					onClick={onToggleDebug}
					title="Settings / API debugger"
					aria-label="Settings"
					className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface-solid text-muted transition-all duration-150 hover:text-ink active:scale-95"
				>
					<Cpu className="h-4 w-4" />
				</button>
			</div>

			{/* Active session name is shown in the tab itself; keep title accessible */}
			<span className="sr-only">{activeSession.name}</span>
		</div>
	);
}
