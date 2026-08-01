def test_engine_event_is_persisted_in_mock_mongo(mock_mongo_api):
    client, events = mock_mongo_api
    marker = "C2.3-MOCK-ONLY"
    payload = {
        "timestamp": "2026-07-28T00:00:00Z",
        "sensorId": marker,
        "species": "Koala",
        "microphoneLLA": [-37.8136, 144.9631, 10.0],
        "animalEstLLA": [-37.8135, 144.9632, 10.0],
        "animalTrueLLA": [-37.8134, 144.9633, 10.0],
        "animalLLAUncertainty": 5,
        "audioClip": "mock-only-no-real-audio",
        "confidence": 95.0,
        "sampleRate": 48000,
    }

    response = client.post("/engine/event", json=payload)

    assert response.status_code == 201
    response_event = response.json()
    stored_event = events.find_one({"sensorId": marker})

    assert stored_event is not None
    assert events.count_documents({"sensorId": marker}) == 1
    assert str(stored_event["_id"]) == response_event["_id"]
    assert stored_event["species"] == "Koala"
    assert stored_event["confidence"] == 95.0
