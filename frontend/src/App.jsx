import React, { useState } from "react";
import axios from "axios";
import Spline from "@splinetool/react-spline";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [showAnalytics, setShowAnalytics] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewImage(URL.createObjectURL(file));
      setColorizedImage(null);
      setErrorMsg("");
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setErrorMsg("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      setLoading(true);
      setErrorMsg("");

      const response = await axios.post("http://127.0.0.1:8000/upload", formData, {
        responseType: "blob",
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const imageBlob = response.data;
      const imageUrl = URL.createObjectURL(imageBlob);
      setColorizedImage(imageUrl);
    } catch (error) {
      console.error("Upload failed:", error);
      setErrorMsg("Upload failed. Check your backend or try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden text-white font-sans">
      {/* Background Spline */}
      <div className="absolute inset-0 z-0 bg-black">
        <Spline scene="https://prod.spline.design/6aIx6pXYtajhU59A/scene.splinecode" />
      </div>

      {/* Main UI */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-10">
        <div className="w-full max-w-3xl bg-black/50 border border-white/20 backdrop-blur-md rounded-3xl p-10 shadow-2xl">
          <h1 className="text-4xl sm:text-5xl font-extrabold text-center text-white drop-shadow-lg mb-4 tracking-tight">
            SAR Colorization Studio
          </h1>
          <p className="text-center text-white text-md sm:text-lg mb-6 drop-shadow">
            AI-powered enhancement for grayscale{" "}
            <span className="text-indigo-300 font-semibold">Synthetic Aperture Radar</span> images.
            Inspired by <span className="text-indigo-300 font-semibold">ISRO</span>, made for the future.
          </p>

          <div className="flex flex-col gap-4 items-center justify-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-100 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:font-semibold file:bg-[#8D7FC5] file:text-white hover:file:bg-[#CDC1FF] transition"
            />
            <button
              onClick={handleUpload}
              disabled={loading}
              className="bg-[#CDC1FF] hover:bg-[#F5EFFF] text-[#2E073F] font-bold py-2 px-6 rounded-full shadow-lg transition-all duration-200 hover:scale-105 disabled:opacity-50"
            >
              {loading ? "Colorizing..." : "✨ Upload & Enhance"}
            </button>
          </div>

          {errorMsg && (
            <div className="mt-4 text-red-400 text-center animate-pulse">
              🚫 {errorMsg}
            </div>
          )}

          {previewImage && (
            <div className="mt-10">
              <h2 className="text-2xl font-bold text-center text-white mb-4 drop-shadow">
                Uploaded Input Image
              </h2>
              <div className="w-full flex justify-center">
                <img
                  src={previewImage}
                  alt="Uploaded SAR"
                  className="rounded-2xl border-4 border-white/30 shadow-xl max-w-full max-h-[400px] object-contain"
                />
              </div>
            </div>
          )}

          {colorizedImage && (
            <div className="mt-10">
              <h2 className="text-2xl font-bold text-center text-white mb-4 drop-shadow">
                Your Colorized Image
              </h2>
              <div className="w-full flex justify-center">
                <img
                  src={colorizedImage}
                  alt="Colorized SAR"
                  className="rounded-2xl border-4 border-[#CDC1FF] shadow-2xl max-w-full max-h-[500px] object-contain transition-all duration-300"
                />
              </div>
              <div className="mt-6 text-center">
                <a
                  href={colorizedImage}
                  download="colorized_output.png"
                  className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-6 rounded-full transition duration-200 hover:scale-105"
                >
                  Download Result
                </a>
              </div>
            </div>
          )}

          {/* Analytics Button */}
          <div className="mt-6 text-center">
  <button
    onClick={() => setShowAnalytics(!showAnalytics)}
    className="bg-indigo-500 hover:bg-indigo-600 text-white font-bold py-2 px-6 rounded-full transition duration-200 hover:scale-105"
  >
    {showAnalytics ? "Hide Analytics" : "📊 View Color Analytics"}
  </button>
</div>
        </div>
      </div>

      {/* Analytics Modal */}
      {showAnalytics && (
  <div className="mt-6 bg-white/10 p-6 rounded-xl text-sm text-white border border-white/20 backdrop-blur-md">
    <h3 className="text-lg font-semibold text-indigo-200 mb-4">Color Interpretation</h3>
    <ul className="list-disc pl-6 space-y-2 text-left">
      <li><span className="font-bold text-green-300">Green</span> — Vegetation / Forests</li>
      <li><span className="font-bold text-yellow-200">Brown / Tan</span> — Barren land / Dry soil</li>
      <li><span className="font-bold text-blue-300">Blue / Dark Blue</span> — Rivers, Lakes, Oceans</li>
      <li><span className="font-bold text-gray-200">Gray / White</span> — Urban areas / Cloud cover</li>
      <li><span className="font-bold text-purple-300">Purple / Deep Red</span> — Dense vegetation / Croplands</li>
    </ul>
  </div>
)}
    </div>
  );
  
}

export default App;
