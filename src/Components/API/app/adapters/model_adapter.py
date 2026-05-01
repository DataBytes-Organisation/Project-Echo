import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class AdapterError(Exception):
    pass


class AdapterTimeoutError(AdapterError):
    pass


@dataclass
class PredictionResult:
    species: str
    confidence: float
    raw: Optional[Dict[str, Any]] = None


class HttpModelAdapter:
    def __init__(
        self,
        model_server_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        signature_name: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model_server_url = model_server_url or os.getenv(
            "MODEL_SERVER_URL",
            "http://ts-echo-model-cont:8501/v1/models/echo_model/versions/1:predict",
        )
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", "10")
        )
        self.signature_name = signature_name or os.getenv(
            "MODEL_SIGNATURE_NAME", "serving_default"
        )
        self.class_names = class_names or self._parse_class_names(
            os.getenv("MODEL_CLASS_NAMES", "")
        )

    @staticmethod
    def _parse_class_names(value: str) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def predict(self, model_inputs: Any) -> PredictionResult:
        payload = {
            "signature_name": self.signature_name,
            "inputs": model_inputs,
        }

        try:
            response = requests.post(
                self.model_server_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise AdapterTimeoutError("Model server request timed out") from exc
        except requests.RequestException as exc:
            raise AdapterError(f"Failed to connect model server: {exc}") from exc

        if response.status_code >= 400:
            raise AdapterError(
                f"Model server returned status {response.status_code}: {response.text}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise AdapterError("Model server returned non-JSON response") from exc

        return self._map_response(body)

    def _map_response(self, body: Dict[str, Any]) -> PredictionResult:
        vector = self._extract_scores(body)
        if not vector:
            raise AdapterError("Model server response does not contain prediction scores")

        max_index = max(range(len(vector)), key=lambda idx: vector[idx])
        confidence = float(vector[max_index])
        species = (
            self.class_names[max_index]
            if self.class_names and max_index < len(self.class_names)
            else f"class_{max_index}"
        )
        return PredictionResult(species=species, confidence=confidence, raw=body)

    def _extract_scores(self, body: Dict[str, Any]) -> List[float]:
        candidates: List[Any] = []
        if "outputs" in body:
            candidates.append(body["outputs"])
        if "predictions" in body:
            candidates.append(body["predictions"])

        for candidate in candidates:
            vector = self._flatten_first_numeric_vector(candidate)
            if vector:
                return vector
        return []

    def _flatten_first_numeric_vector(self, value: Any) -> List[float]:
        if isinstance(value, list):
            if value and all(isinstance(item, (int, float)) for item in value):
                return [float(item) for item in value]
            for item in value:
                nested = self._flatten_first_numeric_vector(item)
                if nested:
                    return nested
        elif isinstance(value, dict):
            for nested_value in value.values():
                nested = self._flatten_first_numeric_vector(nested_value)
                if nested:
                    return nested
        return []
