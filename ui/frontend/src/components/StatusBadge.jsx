const STATUS_STYLES = {
	ready: {
		label: "System ready",
		className: "text-success bg-success-soft",
		pulse: false,
	},
	"inference-error": {
		label: "Backend online · inference failed",
		className: "text-danger bg-danger-soft",
		pulse: false,
	},
	loading: {
		label: "Checking backend…",
		className: "text-warning bg-warning-soft",
		pulse: true,
	},
	error: {
		label: "Backend error",
		className: "text-danger bg-danger-soft",
		pulse: false,
	},
};

export default function StatusBadge({ status = "ready" }) {
	const style = STATUS_STYLES[status] || STATUS_STYLES.ready;

	return (
		<span
			role="status"
			className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-colors duration-200 ${style.className}`}
		>
			<span
				className={`h-1.5 w-1.5 rounded-full bg-current ${style.pulse ? "animate-pulse" : ""}`}
			/>
			{style.label}
		</span>
	);
}
