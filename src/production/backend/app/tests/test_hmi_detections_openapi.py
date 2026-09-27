"""OpenAPI contract for GET /hmi/detections; fakes Mongo at the import boundary."""
import unittest
from unittest.mock import MagicMock, patch

with patch("pymongo.MongoClient", return_value=MagicMock()):
    from fastapi import FastAPI
    from app.routers import hmi


REQUIRED_FIELDS = {
    "sourceType",
    "timestamp",
    "sensorId",
    "species",
    "microphoneLLA",
    "confidence",
}


class HmiDetectionsOpenApiTests(unittest.TestCase):
    def test_detections_response_schema_is_discoverable_array(self):
        app = FastAPI()
        app.include_router(hmi.router, prefix="/hmi")
        spec = app.openapi()
        response = spec["paths"]["/hmi/detections"]["get"]["responses"]["200"]
        schema = response["content"]["application/json"]["schema"]
        self.assertNotEqual(schema, {}, "response schema is undocumented")
        self.assertEqual(schema.get("type"), "array")
        ref = schema.get("items", {}).get("$ref", "")
        self.assertTrue(ref.startswith("#/components/schemas/"), ref)
        name = ref.rsplit("/", 1)[-1]
        self.assertEqual(name, "RealDetectionRead")
        props = set(spec["components"]["schemas"][name]["properties"])
        self.assertTrue(REQUIRED_FIELDS.issubset(props), props)


if __name__ == "__main__":
    unittest.main()
