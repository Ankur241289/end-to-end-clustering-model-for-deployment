from flask import Flask, request, render_template
import logging
from werkzeug.exceptions import HTTPException

# ML imports (keep as in your project)
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# App setup
application = Flask(__name__, static_folder="static", template_folder="templates")
app = application
app.config["SECRET_KEY"] = "replace-with-a-secure-random-key"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        # Read and sanitize form values
        gender = request.form.get("gender", "").strip()
        ethnicity = request.form.get("ethnicity", "").strip()
        parental_education = request.form.get("parental_level_of_education", "").strip()
        lunch = request.form.get("lunch", "").strip()
        test_prep = request.form.get("test_preparation_course", "").strip()

        reading_score_raw = request.form.get("reading_score", "").strip()
        writing_score_raw = request.form.get("writing_score", "").strip()

        # Validate numeric inputs
        if reading_score_raw == "" or writing_score_raw == "":
            error = "Please provide both reading and writing scores (0–100)."
            logger.warning("Validation error: %s", error)
            return render_template("home.html", error=error, form=request.form)

        reading_score = float(reading_score_raw)
        writing_score = float(writing_score_raw)

        # Validate ranges
        if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
            error = "Scores must be between 0 and 100."
            logger.warning("Validation error: %s", error)
            return render_template("home.html", error=error, form=request.form)

        # Build CustomData object with correct mapping
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_prep,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_data_frame()
        logger.info("Input dataframe for prediction: %s", pred_df.to_dict(orient="records"))

        # Run prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        predicted_value = results[0] if hasattr(results, "__len__") else results
        logger.info("Predicted Mathematics marks: %s", predicted_value)

        return render_template("home.html", results=predicted_value, form=request.form)

    except ValueError as ve:
        logger.exception("ValueError during prediction: %s", ve)
        return render_template("home.html", error="Invalid numeric input. Please enter valid numbers.", form=request.form)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Unexpected error during prediction: %s", e)
        return render_template(
            "home.html",
            error="An unexpected error occurred. Please try again later.",
            form=request.form,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)