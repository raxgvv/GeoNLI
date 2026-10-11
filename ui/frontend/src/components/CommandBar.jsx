import { Send, Target, HelpCircle, MapPin, X } from "lucide-react";

const QUICK_PROMPTS = {
	grounding: ["Where are the buildings?", "Find water bodies", "Locate the roads"],
	vqa: {
		binary: ["Is there a river?", "Are there vehicles present?"],
		numeric: ["How many ships?", "How many buildings are there?"],
		semantic: ["What is the land use here?", "Describe the terrain"],
	},
};

const VQA_OPTIONS = [
	{ id: "binary", label: "Binary" },
	{ id: "numeric", label: "Numeric" },
	{ id: "semantic", label: "Semantic" },
];

export default function CommandBar({
	inputMessage,
	setInputMessage,
	onSendMessage,
	isLoading,
	isImageLoaded,
	selectedTask,
	setSelectedTask,
	vqaType,
	setVqaType,
	messages,
	toast,
	onDismissToast,
}) {
	const getPlaceholder = () => {
		if (!isImageLoaded) return "Upload image first...";
		if (selectedTask === "grounding")
			return "Describe object to locate (e.g. 'Find water bodies')...";
		if (selectedTask === "vqa") {
			if (vqaType === "binary") return "Ask a Yes/No question (e.g. 'Is there a river?')...";
			if (vqaType === "numeric") return "Ask for a count/number (e.g. 'How many buildings?')...";
			return "Ask a general question about the scene...";
		}
		return "Enter your query...";
	};

	const quickPrompts =
		selectedTask === "grounding"
			? QUICK_PROMPTS.grounding
			: QUICK_PROMPTS.vqa[vqaType];

	return (
		<div className="ui-layer fixed bottom-4 left-1/2 z-30 w-[min(760px,calc(100vw-24px))] -translate-x-1/2">
			<div className="flex flex-col items-stretch gap-2">
				{/* Answer toast — only when the log sheet is closed */}
				{toast && (
					<div className="mx-auto flex max-w-full items-start gap-2 rounded-[16px] border border-line bg-surface px-3 py-2 text-xs text-ink shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] animate-rise-in">
						<MapPin className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent-text" />
						<p className="line-clamp-2 min-w-0 flex-1">{toast}</p>
						<button
							type="button"
							onClick={onDismissToast}
							aria-label="Dismiss"
							className="shrink-0 rounded-full p-0.5 text-muted transition-colors hover:text-ink"
						>
							<X className="h-3.5 w-3.5" />
						</button>
					</div>
				)}

				{/* Quick chips — only before the first message */}
				{messages.length === 0 && (
					<div className="flex flex-wrap justify-center gap-2">
						{quickPrompts.map(prompt => (
							<button
								key={prompt}
								type="button"
								disabled={!isImageLoaded}
								title={isImageLoaded ? "Fill the input with this prompt" : "Upload an image first"}
								onClick={() => setInputMessage(prompt)}
								className="rounded-full border border-line bg-surface px-3 py-1.5 text-xs text-muted shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] transition-colors duration-150 hover:border-accent/50 hover:text-accent-text active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50"
							>
								{prompt}
							</button>
						))}
					</div>
				)}

				{/* VQA subtype pills — expand above the bar */}
				{selectedTask === "vqa" && (
					<div className="flex justify-center animate-expand-in">
						<div
							role="group"
							aria-label="VQA type"
							className="flex items-center gap-1 rounded-full border border-line bg-surface p-1 shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)]"
						>
							{VQA_OPTIONS.map(opt => (
								<button
									key={opt.id}
									type="button"
									aria-pressed={vqaType === opt.id}
									onClick={() => setVqaType(opt.id)}
									className={`rounded-full px-3 py-1 text-xs font-medium transition-colors duration-150 ${
										vqaType === opt.id
											? "bg-accent-fill text-on-accent"
											: "text-muted hover:text-ink"
									}`}
								>
									{opt.label}
								</button>
							))}
						</div>
					</div>
				)}

				{/* The bar */}
				<form
					onSubmit={onSendMessage}
					className="flex items-center gap-2 rounded-full border border-line bg-surface p-2 pl-3 shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)]"
				>
					{/* Mode switch */}
					<div
						role="group"
						aria-label="Task mode"
						className="flex shrink-0 items-center gap-0.5 rounded-full bg-line/35 p-0.5"
					>
						<button
							type="button"
							aria-pressed={selectedTask === "grounding"}
							onClick={() => setSelectedTask("grounding")}
							title="Grounding mode"
							className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition-colors duration-150 ${
								selectedTask === "grounding"
									? "bg-accent-fill text-on-accent shadow-sm"
									: "text-muted hover:text-ink"
							}`}
						>
							<Target className="h-3.5 w-3.5" />
							<span className="hidden sm:inline">Grounding</span>
						</button>
						<button
							type="button"
							aria-pressed={selectedTask === "vqa"}
							onClick={() => setSelectedTask("vqa")}
							title="VQA mode"
							className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition-colors duration-150 ${
								selectedTask === "vqa"
									? "bg-accent-fill text-on-accent shadow-sm"
									: "text-muted hover:text-ink"
							}`}
						>
							<HelpCircle className="h-3.5 w-3.5" />
							<span className="hidden sm:inline">VQA</span>
						</button>
					</div>

					<input
						type="text"
						value={inputMessage}
						onChange={e => setInputMessage(e.target.value)}
						placeholder={getPlaceholder()}
						disabled={isLoading || !isImageLoaded}
						aria-label="Message"
						title={
							!isImageLoaded
								? "Upload an image first"
								: isLoading
									? "Waiting for the model response"
									: undefined
						}
						className="min-w-0 flex-1 border-none bg-transparent px-1 py-1.5 text-sm text-ink placeholder:text-muted focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
					/>

					<button
						type="submit"
						disabled={isLoading || !isImageLoaded || !inputMessage.trim()}
						title={
							!isImageLoaded
								? "Upload an image first"
								: isLoading
									? "Waiting for the model response"
									: !inputMessage.trim()
										? "Enter a message"
										: "Send"
						}
						aria-label="Send message"
						className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-fill text-on-accent transition-all duration-150 hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
					>
						<Send className="h-4 w-4" />
					</button>
				</form>
			</div>
		</div>
	);
}
