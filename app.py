from web.app_factory import create_app

# Entry point for VS Code "Run Python File" or `python app.py`
app = create_app()

if __name__ == "__main__":
    # Use debug=True for dev; in prod, use a WSGI server.
    app.run(host="127.0.0.1", port=5000, debug=True)
