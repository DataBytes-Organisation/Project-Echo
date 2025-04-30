import React, { useEffect, useState } from "react";

const SubmissionOverview = () => {
  const [submissions, setSubmissions] = useState([]);

  useEffect(() => {
    fetch("submissions.json")
      .then((res) => res.json())
      .then((data) => setSubmissions(data));
  }, []);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Submission Overview</h1>

      {submissions.length === 0 ? (
        <p className="text-gray-500">No submissions yet.</p>
      ) : (
        <div className="space-y-4">
          {submissions.map((submission) => (
            <div
              key={submission.id}
              className="bg-white shadow rounded-xl p-4 border border-gray-200"
            >
              <div className="flex justify-between items-center mb-2">
                <div>
                  <p className="font-semibold text-lg">
                    🎧 {submission.fileName}
                  </p>
                  <p className="text-sm text-gray-500">
                    Submitted on: {submission.date}
                  </p>
                </div>
                <audio controls className="w-48">
                  <source src={submission.audioUrl} type="audio/mpeg" />
                  Your browser does not support the audio element.
                </audio>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-600 mb-1">
                  Detected Animals:
                </p>
                <div className="flex flex-wrap gap-2">
                  {submission.detectedAnimals.map((animal, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm"
                    >
                      🐾 {animal}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SubmissionOverview;
