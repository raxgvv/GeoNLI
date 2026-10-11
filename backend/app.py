import logging
import os
import time
from uuid import uuid4
from pathlib import Path

from flask import Flask, g, has_request_context, request
from flask_cors import CORS
from backend.config import Config
from backend.api.routes import api_bp
from backend.services.request_context import REQUEST_ID

def _configure_logging():
    """Send backend logs to both stdout and geonli.log for parity with FastAPI."""
    log_path = os.environ.get(
        "GEONLI_LOG",
        str(Path(__file__).resolve().parents[1] / "geonli.log"),
    )
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler  
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    # Set levels and add handlers
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    # Ensure backend module loggers inherit the level
    logging.getLogger('backend').setLevel(logging.INFO)
    logging.getLogger('models').setLevel(logging.INFO)
    
    print("✅ Logging configured - INFO level messages will appear in terminal and geonli.log")

def create_app():
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 GeoNLI Backend starting up...")
    app = Flask(__name__)
    
    # Load Config
    Config.init_app(app)
    app.config.from_object(Config)
    
    # Enable CORS (Critical for Frontend access)
    CORS(app)

    @app.before_request
    def _start_request_timing():
        from flask import g
        g.request_started = time.monotonic()
        g.request_id = uuid4().hex[:12]
        g.request_id_token = REQUEST_ID.set(g.request_id)
        logging.getLogger('backend.http').info(
            "Request started | request_id=%s | method=%s | path=%s",
            g.request_id, request.method, request.path,
        )

    @app.after_request
    def _log_response_status(response):
        logging.getLogger('backend.http').info(
            "HTTP response | request_id=%s | method=%s | path=%s | status=%s | total_s=%.3f",
            getattr(g, 'request_id', '-'),
            request.method if has_request_context() else 'unknown',
            request.path if has_request_context() else 'unknown',
            response.status_code,
            time.monotonic() - getattr(g, 'request_started', time.monotonic()) if has_request_context() else 0.0,
        )
        return response

    @app.teardown_request
    def _clear_request_context(_error):
        from flask import g
        token = getattr(g, 'request_id_token', None)
        if token is not None:
            REQUEST_ID.reset(token)
    
    # Register Blueprints
    app.register_blueprint(api_bp, url_prefix='/geoNLI')
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    logger = logging.getLogger(__name__)
    host = os.environ.get('BACKEND_HOST', '0.0.0.0')
    logger.info(f"🌐 Starting GeoNLI server on http://{host}:{Config.PORT} (debug={Config.DEBUG})")
    
    # Run on 0.0.0.0 to be accessible externally (Required for Evaluation)
    app.run(
        host=host,
        port=Config.PORT,
        debug=Config.DEBUG,
        use_reloader=False,
    )
