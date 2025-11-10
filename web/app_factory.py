from flask import Flask
from flask_cors import CORS

from config import Config
from .routes import register_routes

def create_app():
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.config.from_object(Config)
    CORS(app)

    # Register blueprints / routes
    register_routes(app)

    return app
