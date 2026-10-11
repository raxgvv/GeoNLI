import { Eye, EyeOff } from "lucide-react";

export default function HideUiToggle({ hidden, onToggle }) {
	return (
		<button
			type="button"
			onClick={onToggle}
			aria-pressed={hidden}
			aria-label={hidden ? "Show UI" : "Hide UI"}
			title={hidden ? "Show UI (H)" : "Hide UI (H)"}
			className="fixed top-2.5 right-3 z-50 flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface text-muted shadow-[var(--float-shadow)] backdrop-blur-[var(--blur)] transition-all duration-150 hover:text-ink active:scale-95"
		>
			{hidden ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
		</button>
	);
}
