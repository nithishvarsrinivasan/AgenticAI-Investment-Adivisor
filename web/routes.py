from flask import Blueprint, request, jsonify, render_template
from services.chatbot import MLChatbot
from services.crewai_pipeline import run_investment_crew

bp = Blueprint("core", __name__)
ml_chatbot = MLChatbot()

@bp.route("/")
def index():
    return render_template("index.html")

@bp.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True) or {}
        stock_symbol = (data.get("symbol") or "").strip().upper()
        if not stock_symbol:
            return jsonify({"error": "No stock symbol provided"}), 400
        analysis_html, recommendation_html = run_investment_crew(stock_symbol)
        return jsonify({
            "analysis": analysis_html,
            "recommendation": recommendation_html
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/train_model", methods=["POST"])
def train_model():
    try:
        data = request.get_json(force=True) or {}
        analysis = data.get("analysis", "")
        recommendation = data.get("recommendation", "")
        if not analysis or not recommendation:
            return jsonify({"error": "No analysis data provided"}), 400
        ok = ml_chatbot.train(analysis, recommendation)
        if ok:
            return jsonify({
                "message": f"✅ Model trained successfully! I analyzed {len(ml_chatbot.sentences)} sentences from the report. You can now ask me questions!",
                "sentences_count": len(ml_chatbot.sentences),
            })
        return jsonify({"error": "Training failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/ask_question", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(force=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        if not ml_chatbot.trained:
            return jsonify({"error": "Model not trained. Please train first."}), 400
        answer = ml_chatbot.answer_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def register_routes(app):
    app.register_blueprint(bp)
