// Disable default body parser to handle multipart data manually
export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const GPU_SERVER_URL = globalThis.process?.env?.VITE_BACKEND_BASE_URL;
    const targetUrl = `${GPU_SERVER_URL}/geoNLI/eval`;
    
    console.log(`[VERCEL PROXY] Forwarding to: ${targetUrl}`);
    console.log(`[VERCEL PROXY] Content-Type:`, req.headers['content-type']);
    
    // Forward the raw request body and headers
    const response = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'content-type': req.headers['content-type'],
        'content-length': req.headers['content-length'],
      },
      body: req,
      duplex: 'half',
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[VERCEL PROXY] GPU server error:`, response.status, errorText);
      return res.status(response.status).json({
        error: `GPU server error: ${response.status}`,
        details: errorText
      });
    }
    
    // Parse JSON response
    const data = await response.json();
    return res.status(response.status).json(data);
    
  } catch (error) {
    console.error('[VERCEL PROXY] Error:', error.message);
    return res.status(502).json({
      error: 'GPU server unavailable',
      details: error.message
    });
  }
}
