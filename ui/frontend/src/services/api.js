/**
 * API Service for GeoNLI
 * Handles communication with the Python/GPU backend or simulates it.
 */

// Set VITE_BACKEND_BASE_URL for another host; local development targets Flask.
export const BACKEND_BASE_URL = (import.meta.env.VITE_BACKEND_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "")

if (typeof window !== 'undefined') {
  console.info('[GeoNLI] Backend base URL:', BACKEND_BASE_URL);
}

export const analyzeImage = async (
	taskType,
	query,
	imageFile,
	isSimulating,
	vqaType = "semantic",
	imageType = "RGB",
	spatialResolutionM = 1.0
) => {
	if (isSimulating) {
		return simulateBackendResponse(taskType, query, vqaType, imageType);
	} else {
		return fetchRealBackend(taskType, query, imageFile, vqaType, imageType, spatialResolutionM);
	}
};

const fetchRealBackend = async (
	taskType,
	query,
	imageFile,
	vqaType = "semantic",
	imageType = "RGB",
	spatialResolutionM = 1.0
) => {
	const controller = new AbortController();
	const timeout = window.setTimeout(() => controller.abort(), 90000);
	try {
		const formData = new FormData();
		formData.append("image", imageFile);
		formData.append("task_type", taskType);
		formData.append("image_type", imageType); // Add Image Type to payload
		formData.append("spatial_resolution_m", spatialResolutionM.toString()); // Add Spatial Resolution to payload

		if (taskType === "vqa") {
			formData.append("vqa_type", vqaType);
		}
		formData.append("query", taskType === "captioning" ? "" : query);

		const response = await fetch(`${BACKEND_BASE_URL}/geoNLI/eval`, {
			method: "POST",
			body: formData,
			signal: controller.signal,
		});
		const payload = await response.json().catch(() => ({}));
		if (!response.ok || payload.success === false) {
			const requestRef = payload.request_id ? ` (request ${payload.request_id})` : "";
			const message = `${payload.error || payload.details || `HTTP ${response.status}`}${requestRef}`;
			throw new Error(`Request failed (HTTP ${response.status}): ${message}`);
		}
		if (!payload.answer && taskType !== "grounding") {
			throw new Error("Backend returned an empty answer.");
		}
		return payload;
	} catch (error) {
		console.error("API Error:", error);
		if (error.name === "AbortError") throw new Error("Request timed out after 90 seconds. The backend may still be finishing this inference; use the request ID in server logs if provided, then retry.");
		if (error instanceof TypeError) throw new Error(`Cannot reach backend at ${BACKEND_BASE_URL}. Check that Flask is running.`);
		throw error;
	} finally {
		window.clearTimeout(timeout);
	}
};

export const checkBackendHealth = async () => {
	const response = await fetch(`${BACKEND_BASE_URL}/geoNLI/health`, { signal: AbortSignal.timeout(5000) });
	if (!response.ok) throw new Error(`Health check failed (HTTP ${response.status})`);
	const payload = await response.json();
	if (payload.status !== "active") throw new Error("Backend is not active");
	return true;
};

const simulateBackendResponse = async (taskType, query, vqaType, imageType) => {
	return new Promise(resolve => {
		setTimeout(() => {
			let response = {
				answer: "I processed your request.",
				bboxes: [],
				confidence: 0.98,
			};

			if (taskType === "captioning") {
				response.answer = `[${imageType} Image Analysis] The image captures a high-resolution satellite view. It features dense residential clusters interspersed with vegetation.`;
			} else if (taskType === "grounding") {
				response.answer = `I have located the requested objects based on "${query}".`;
				if (query.toLowerCase().includes("water")) {
					// Normalized 8-point format: (x1, y1, x2, y2, x3, y3, x4, y4)
					// Example: A vertical strip on the right
					response.bboxes = [[0.7, 0.0, 1.0, 0.0, 1.0, 1.0, 0.7, 1.0]];
				} else {
					// Example: Two arbitrary quadrilaterals (houses/regions)
					response.bboxes = [
						// Box 1 (Top Leftish)
						[0.1, 0.1, 0.4, 0.1, 0.4, 0.35, 0.1, 0.35],
						// Box 2 (Bottom Center - Slightly rotated look)
						[0.5, 0.6, 0.7, 0.65, 0.65, 0.85, 0.45, 0.8],
					];
				}
			} else {
				// VQA
				const lowerQuery = query.toLowerCase();
				if (vqaType === "numeric" || lowerQuery.includes("count")) {
					response.answer = "45"; // Numeric response
				} else if (vqaType === "binary") {
					response.answer = "Yes"; // Binary response
				} else {
					response.answer = `Here is the answer to your question regarding the ${imageType} imagery: "${query}".`;
				}
			}
			resolve(response);
		}, 1500);
	});
};
