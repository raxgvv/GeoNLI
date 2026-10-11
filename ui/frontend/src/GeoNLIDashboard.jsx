import { useState, useEffect, useRef } from "react";
import Stage from "./components/Stage";
import SessionStrip from "./components/SessionStrip";
import SceneInspector from "./components/SceneInspector";
import CommandBar from "./components/CommandBar";
import LogSheet from "./components/LogSheet";
import CaptionCard from "./components/CaptionCard";
import HideUiToggle from "./components/HideUiToggle";
import DebugPanel from "./components/DebugPanel";
import ConfirmDialog from "./components/ConfirmDialog";
import { analyzeImage, checkBackendHealth } from "./services/api";

const LOG_AUTO_CLOSE_MS = 6000;
const TOAST_MS = 10000;

export default function GeoNLIDashboard() {
	const [sessions, setSessions] = useState([
		{ id: 1, name: "Sample Region 01", image: null, messages: [], caption: "" },
	]);
	const [activeSessionId, setActiveSessionId] = useState(1);
	const [inputMessage, setInputMessage] = useState("");
	const [selectedTask, setSelectedTask] = useState("grounding");
	const [vqaType, setVqaType] = useState("semantic");
	const [imageType, setImageType] = useState("RGB");
	const [spatialResolution, setSpatialResolution] = useState(1.0);
	const [isSimulating, setIsSimulating] = useState(false);
	const [isLoading, setIsLoading] = useState(false);
	const [isGeneratingCaption, setIsGeneratingCaption] = useState(false);
	const [showDebug, setShowDebug] = useState(false);
	const [isDarkMode, setIsDarkMode] = useState(false);
	const [apiStatus, setApiStatus] = useState("loading");
	const [terminateConfirm, setTerminateConfirm] = useState(null);
	const sessionsRef = useRef(sessions);
	sessionsRef.current = sessions;

	// Presentation-only state (no functional impact)
	const [uiHidden, setUiHidden] = useState(false);
	const [logOpen, setLogOpen] = useState(false);
	const [logPinned, setLogPinned] = useState(false);
	const [answerToast, setAnswerToast] = useState(null);

	const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];
	const messages = activeSession.messages;
	const lastBotMessage = [...messages].reverse().find(m => m.role === "bot");
	const activeBBoxes = lastBotMessage?.bboxes || [];

	const logAutoCloseTimer = useRef(null);
	const toastTimer = useRef(null);
	const prevMsgCount = useRef(messages.length);
	const requestInFlight = useRef(false);

	const clearLogTimer = () => {
		if (logAutoCloseTimer.current) {
			clearTimeout(logAutoCloseTimer.current);
			logAutoCloseTimer.current = null;
		}
	};
	const clearToastTimer = () => {
		if (toastTimer.current) {
			clearTimeout(toastTimer.current);
			toastTimer.current = null;
		}
	};

	// Mirror the theme class on <html> so base styles, scrollbars and
	// native controls follow the active theme.
	useEffect(() => {
		const root = document.documentElement;
		root.classList.toggle("dark", isDarkMode);
		root.classList.toggle("light", !isDarkMode);
	}, [isDarkMode]);

	useEffect(() => {
		return () => {
			clearLogTimer();
			clearToastTimer();
			sessionsRef.current.forEach(session => {
				if (session.image && session.image.startsWith("blob:")) {
					URL.revokeObjectURL(session.image);
				}
			});
		};
	}, []);

	useEffect(() => {
		let mounted = true;
		const check = async () => {
			try {
				await checkBackendHealth();
				if (mounted) setApiStatus(current => current === "error" || current === "loading" ? "ready" : current);
			} catch (error) {
				console.warn("GeoNLI backend health check failed:", error);
				if (mounted) setApiStatus("error");
			}
		};
		check();
		const timer = window.setInterval(check, 30000);
		return () => { mounted = false; window.clearInterval(timer); };
	}, []);

	// New messages: open the log sheet (auto-close unless pinned) and show a
	// compact answer toast near the command bar while the sheet is closed.
	useEffect(() => {
		const count = messages.length;
		if (count < prevMsgCount.current) {
			prevMsgCount.current = count;
			return;
		}
		if (count > prevMsgCount.current) {
			prevMsgCount.current = count;
			const last = messages[count - 1];

			setLogOpen(true);
			clearLogTimer();
			if (!logPinned) {
				logAutoCloseTimer.current = setTimeout(() => setLogOpen(false), LOG_AUTO_CLOSE_MS);
			}

			if (last.role === "bot") {
				setAnswerToast(last.content);
				clearToastTimer();
				toastTimer.current = setTimeout(() => setAnswerToast(null), TOAST_MS);
			}
		}
	}, [messages, logPinned]);

	// Hide/show all floating UI with the H key
	useEffect(() => {
		const onKeyDown = e => {
			if (e.key?.toLowerCase() !== "h") return;
			if (e.metaKey || e.ctrlKey || e.altKey) return;
			const el = e.target;
			const tag = el?.tagName?.toLowerCase();
			if (tag === "input" || tag === "textarea" || tag === "select" || el?.isContentEditable)
				return;
			setUiHidden(v => !v);
		};
		window.addEventListener("keydown", onKeyDown);
		return () => window.removeEventListener("keydown", onKeyDown);
	}, []);

	const handleCreateSession = () => {
		const newId = sessions.length + 1;
		const newSession = {
			id: newId,
			name: `Analysis ${newId}`,
			image: null,
			messages: [],
			caption: "",
		};
		setSessions([...sessions, newSession]);
		setActiveSessionId(newId);
	};

	const handleTerminateSession = sessionId => {
		const session = sessions.find(s => s.id === sessionId);
		if (!session) return;
		setTerminateConfirm(session);
	};

	const confirmTerminate = () => {
		if (!terminateConfirm) return;
		const sessionId = terminateConfirm.id;

		if (terminateConfirm.image && terminateConfirm.image.startsWith("blob:")) {
			URL.revokeObjectURL(terminateConfirm.image);
		}

		setSessions(current => {
			const remaining = current.filter(s => s.id !== sessionId);
			if (remaining.length === 0) {
				const fresh = {
					id: 1,
					name: "Analysis 1",
					image: null,
					messages: [],
					caption: "",
				};
				setActiveSessionId(1);
				return [fresh];
			}
			if (sessionId === activeSessionId) {
				const idx = current.findIndex(s => s.id === sessionId);
				const next = remaining[Math.min(idx, remaining.length - 1)];
				setActiveSessionId(next.id);
			}
			return remaining;
		});

		setTerminateConfirm(null);
	};

	const handleFileUpload = file => {
		if (!file.type.match("image.*")) {
			alert("Please upload a valid image file (PNG, JPG)");
			return;
		}
		const imageUrl = URL.createObjectURL(file);
		setSessions(current =>
			current.map(s =>
				s.id === activeSessionId
					? {
							...s,
							image: imageUrl,
							imageFile: file,
							name: file.name.substring(0, 20),
							messages: [],
							caption: "",
					  }
					: s
			)
		);
	};

	const handleGenerateCaption = async () => {
		if (!activeSession.image || requestInFlight.current) return;
		requestInFlight.current = true;
		setIsGeneratingCaption(true);
		setApiStatus("loading");
		try {
			const data = await analyzeImage(
				"captioning",
				"",
				activeSession.imageFile,
				isSimulating,
				"semantic",
				imageType,
				spatialResolution
			);
			setSessions(current =>
				current.map(s => (s.id === activeSessionId ? { ...s, caption: data.answer } : s))
			);
			setApiStatus("ready");
		} catch (error) {
			console.error(error);
			setApiStatus(error.name === "TypeError" ? "error" : "inference-error");
			alert(`Failed to generate caption: ${error.message || "Unknown error"}`);
		} finally {
			requestInFlight.current = false;
			setIsGeneratingCaption(false);
		}
	};

	const handleSendMessage = async e => {
		e.preventDefault();
		if (!inputMessage.trim() || requestInFlight.current) return;
		if (!activeSession.image) {
			alert("Please upload an EO image first.");
			return;
		}

		const userMsg = {
			role: "user",
			content: inputMessage,
			task: selectedTask,
			subtask: selectedTask === "vqa" ? vqaType : null,
			timestamp: new Date().toISOString(),
		};

		setSessions(current =>
			current.map(s =>
				s.id === activeSessionId ? { ...s, messages: [...s.messages, userMsg] } : s
			)
		);

		setInputMessage("");
		requestInFlight.current = true;
		setIsLoading(true);
		setApiStatus("loading");

		try {
			const data = await analyzeImage(
				selectedTask,
				userMsg.content,
				activeSession.imageFile,
				isSimulating,
				vqaType,
				imageType,
				spatialResolution
			);

			const botMsg = {
				role: "bot",
				content: data.answer || (selectedTask === "grounding" ? "Grounding completed." : ""),
				bboxes: data.bboxes || [],
				timestamp: data.timestamp || new Date().toISOString(),
			};

			setSessions(current =>
				current.map(s =>
					s.id === activeSessionId ? { ...s, messages: [...s.messages, botMsg] } : s
				)
			);
			setApiStatus("ready");
		} catch (error) {
			console.error(error);
			setApiStatus(error.name === "TypeError" ? "error" : "inference-error");
			const errorMsg = {
				role: "bot",
				content: error.message || "Image analysis failed. See the browser console for details.",
				timestamp: new Date().toISOString(),
				isError: true,
			};
			setSessions(current =>
				current.map(s =>
					s.id === activeSessionId ? { ...s, messages: [...s.messages, errorMsg] } : s
				)
			);
		} finally {
			requestInFlight.current = false;
			setIsLoading(false);
		}
	};

	return (
		<div className={isDarkMode ? "dark" : ""}>
			<div
				className={`fixed inset-0 overflow-hidden bg-backdrop font-sans text-ink transition-colors duration-200 ${
					uiHidden ? "ui-hidden" : ""
				}`}
			>
				{/* Stage: full-viewport image layer */}
				<Stage
					image={activeSession.image}
					bboxes={activeBBoxes}
					onFileDrop={handleFileUpload}
				/>

				{/* Floating layer */}
			<SessionStrip
				sessions={sessions}
				activeSessionId={activeSessionId}
				onSessionChange={setActiveSessionId}
				onCreateSession={handleCreateSession}
				onTerminateSession={handleTerminateSession}
				activeSession={activeSession}
				isDarkMode={isDarkMode}
				onToggleTheme={() => setIsDarkMode(!isDarkMode)}
				onToggleDebug={() => setShowDebug(!showDebug)}
				apiStatus={apiStatus}
				logOpen={logOpen}
				onToggleLog={() => {
					clearLogTimer();
					setLogOpen(open => !open);
				}}
				messageCount={messages.length}
			/>

				<SceneInspector
					imageType={imageType}
					setImageType={setImageType}
					spatialResolution={spatialResolution}
					setSpatialResolution={setSpatialResolution}
					isSimulating={isSimulating}
					setIsSimulating={setIsSimulating}
					image={activeSession.image}
					onUpload={handleFileUpload}
				/>

				<CaptionCard
					caption={activeSession.caption}
					image={activeSession.image}
					isGeneratingCaption={isGeneratingCaption}
					onGenerateCaption={handleGenerateCaption}
				/>

				<CommandBar
					inputMessage={inputMessage}
					setInputMessage={setInputMessage}
					onSendMessage={handleSendMessage}
					isLoading={isLoading}
					isImageLoaded={!!activeSession.image}
					selectedTask={selectedTask}
					setSelectedTask={setSelectedTask}
					vqaType={vqaType}
					setVqaType={setVqaType}
					messages={messages}
					toast={!logOpen ? answerToast : null}
					onDismissToast={() => {
						clearToastTimer();
						setAnswerToast(null);
					}}
				/>

				<LogSheet
					open={logOpen}
					onClose={() => {
						clearLogTimer();
						setLogOpen(false);
					}}
					pinned={logPinned}
					onTogglePinned={() => {
						clearLogTimer();
						setLogPinned(p => {
							const next = !p;
							if (next) setLogOpen(true);
							return next;
						});
					}}
					messages={messages}
					isLoading={isLoading}
				/>

				<HideUiToggle hidden={uiHidden} onToggle={() => setUiHidden(v => !v)} />

			<DebugPanel
				isOpen={showDebug}
				onClose={() => setShowDebug(false)}
				info={{
					status: isSimulating ? "MOCK" : "LIVE",
					task: selectedTask.toUpperCase(),
					type: imageType,
					subtask: selectedTask === "vqa" ? vqaType.toUpperCase() : "N/A",
					image: activeSession.imageFile ? activeSession.imageFile.name : "None",
					lastResponse: lastBotMessage,
				}}
			/>

			<ConfirmDialog
				isOpen={!!terminateConfirm}
				title="Terminate Session"
				message={
					terminateConfirm
						? `This will permanently close "${terminateConfirm.name}" and remove all messages, captions, and the uploaded image. This action cannot be undone.`
						: ""
				}
				confirmLabel="Terminate"
				cancelLabel="Cancel"
				onConfirm={confirmTerminate}
				onCancel={() => setTerminateConfirm(null)}
			/>
			</div>
		</div>
	);
}
