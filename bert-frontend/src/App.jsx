import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL;

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzeReview() {
    if (!review.trim()) {
      setError("Please enter a movie review.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: review }),
      });

      if (!response.ok) {
        throw new Error("Unable to analyze review.");
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);

    } catch (err) {
      setError(err.message || "Could not connect to the server.");

    } finally {
      setLoading(false);
    }
  }

  const isPositive = result?.sentiment === "Positive";

  return (
    <main className="container">
      <div className="card">

        <h1>BERT Sentiment Analyzer</h1>

        <p className="subtitle">
          Understand movie reviews using a fine-tuned Transformer model.
        </p>

        <label htmlFor="review">Your movie review</label>

        <textarea
          id="review"
          value={review}
          onChange={(e) => setReview(e.target.value)}
          placeholder="Example: The movie was fantastic..."
        />

        <div className="character-count">
          {review.length} characters
        </div>

        <button onClick={analyzeReview} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>

        {loading && (
          <p className="loading">
            BERT is analyzing your review...
          </p>
        )}

        {error && (
          <p className="error">{error}</p>
        )}

        {result && (
          <section className={`result ${isPositive ? "positive" : "negative"}`}>

            <div className="sentiment-icon">
              {isPositive ? "😊" : "😞"}
            </div>

            <h2>{result.sentiment}</h2>

            <p>Model confidence: {result.confidence}%</p>

            <div className="progress-background">
              <div
                className="progress-fill"
                style={{ width: `${result.confidence}%` }}
              />
            </div>

          </section>
        )}

      </div>
    </main>
  );
}

export default App;