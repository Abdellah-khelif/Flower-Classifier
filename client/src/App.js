import React, { useState } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(
    localStorage.getItem("preview") || null
  );
  const [prediction, setPrediction] = useState(
    localStorage.getItem("prediction") || ""
  );
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
    localStorage.setItem("preview", url);
    setPrediction("");
    localStorage.removeItem("prediction"); // clear previous prediction
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPrediction(data.prediction);
      localStorage.setItem("prediction", data.prediction); // save prediction
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend.");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold mb-6">🌸 Flower Classifier</h1>

      <label className="w-64 h-40 flex flex-col items-center justify-center border-4 border-dashed border-green-400 rounded-lg cursor-pointer hover:bg-green-500 hover:bg-opacity-20 transition duration-300">
        <span className="text-green-400 mb-2">Click or Drag an Image</span>
        <input type="file" className="hidden" onChange={handleFileChange} />
      </label>

      {preview && (
        <img
          src={preview}
          alt="Preview"
          className="w-64 h-64 object-cover rounded-lg mt-6 shadow-lg"
        />
      )}

      <button
        onClick={handlePredict}
        disabled={loading}
        className="mt-6 px-6 py-3 bg-green-500 hover:bg-green-600 rounded-lg text-white font-semibold transition duration-300 disabled:bg-gray-500"
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {prediction && (
        <div className="mt-6 text-2xl font-bold text-green-400">
          Prediction: {prediction}
        </div>
      )}
    </div>
  );
}

export default App;
