from types import SimpleNamespace

from app.routers import audio_upload_router as audio


class FakeStorage:
    pass


class RecordingUploads:
    def __init__(self, fail=False):
        self.fail = fail
        self.documents = []

    def insert_one(self, document):
        if self.fail:
            raise RuntimeError("mongo unavailable")

        self.documents.append(document.copy())
        return SimpleNamespace(
            inserted_id="64b64c1234567890abcdef12"
        )


def post_audio(
    client,
    filename="sample.wav",
    content=b"RIFF-audio",
    content_type="audio/wav",
):
    return client.post(
        "/api/audio/upload",
        files={
            "file": (
                filename,
                content,
                content_type,
            )
        },
        data={"user_id": "test-user"},
    )

def assert_error_response(
    response,
    status_code,
    error_code,
    message,
):
    assert response.status_code == status_code

    body = response.json()

    assert body["status"] == "failed"
    assert body["error"]["code"] == error_code
    assert body["error"]["message"] == message
    assert body["error"]["details"] is None
    assert body["error"]["correlation_id"]

def test_audio_upload_stores_object_in_r2_and_metadata(
    mock_mongo_api,
    monkeypatch,
):
    client, _ = mock_mongo_api

    storage = FakeStorage()
    uploads = RecordingUploads()
    upload_call = {}

    monkeypatch.setattr(
        audio,
        "get_r2_storage",
        lambda: storage,
    )
    monkeypatch.setattr(
        audio,
        "AudioUploads",
        uploads,
    )

    def fake_upload(
        storage_arg,
        source,
        key,
        content_type,
    ):
        upload_call["storage"] = storage_arg
        upload_call["key"] = key
        upload_call["content_type"] = content_type
        upload_call["content"] = source.read()

    monkeypatch.setattr(
        audio,
        "upload_to_r2",
        fake_upload,
    )

    response = post_audio(client)

    assert response.status_code == 200

    body = response.json()

    assert body["message"] == "Upload successful"
    assert body["upload_id"] == "64b64c1234567890abcdef12"
    assert body["storage_key"].startswith("audio_uploads/")
    assert body["storage_key"].endswith(".wav")

    assert upload_call["storage"] is storage
    assert upload_call["content_type"] == "audio/wav"
    assert upload_call["content"] == b"RIFF-audio"
    assert upload_call["key"] == body["storage_key"]

    assert len(uploads.documents) == 1

    metadata = uploads.documents[0]

    assert metadata["original_filename"] == "sample.wav"
    assert metadata["storage_key"] == body["storage_key"]
    assert metadata["content_type"] == "audio/wav"
    assert metadata["size_bytes"] == len(b"RIFF-audio")
    assert metadata["user_id"] == "test-user"

    # Local disk paths are no longer persisted.
    assert "path" not in metadata


def test_r2_unavailable_returns_503(
    mock_mongo_api,
    monkeypatch,
):
    client, _ = mock_mongo_api

    def unavailable():
        raise audio.R2StorageUnavailable(
            "storage unavailable"
        )

    monkeypatch.setattr(
        audio,
        "get_r2_storage",
        unavailable,
    )

    response = post_audio(client)

    assert_error_response(
        response,
        503,
        "SERVICE_UNAVAILABLE",
        "Audio storage is temporarily unavailable",
    )


def test_r2_upload_failure_returns_502(
    mock_mongo_api,
    monkeypatch,
):
    client, _ = mock_mongo_api

    storage = FakeStorage()

    monkeypatch.setattr(
        audio,
        "get_r2_storage",
        lambda: storage,
    )

    def fail_upload(*args, **kwargs):
        raise audio.R2UploadFailed("upload failed")

    monkeypatch.setattr(
        audio,
        "upload_to_r2",
        fail_upload,
    )

    response = post_audio(client)

    assert_error_response(
        response,
        502,
        "UPSTREAM_ERROR",
        "Failed to upload audio to object storage",
    )


def test_mongo_failure_removes_uploaded_r2_object(
    mock_mongo_api,
    monkeypatch,
):
    client, _ = mock_mongo_api

    storage = FakeStorage()
    uploads = RecordingUploads(fail=True)

    uploaded = {}
    deleted = {}

    monkeypatch.setattr(
        audio,
        "get_r2_storage",
        lambda: storage,
    )
    monkeypatch.setattr(
        audio,
        "AudioUploads",
        uploads,
    )

    def successful_upload(
        storage_arg,
        source,
        key,
        content_type,
    ):
        uploaded["key"] = key

    def record_delete(storage_arg, key):
        deleted["storage"] = storage_arg
        deleted["key"] = key
        return True

    monkeypatch.setattr(
        audio,
        "upload_to_r2",
        successful_upload,
    )
    monkeypatch.setattr(
        audio,
        "delete_from_r2_best_effort",
        record_delete,
    )

    response = post_audio(client)

    assert_error_response(
        response,
        500,
        "INTERNAL_ERROR",
        "Failed to save upload metadata",
    )

    assert deleted["storage"] is storage
    assert deleted["key"] == uploaded["key"]


def test_invalid_extension_is_rejected_before_r2(
    mock_mongo_api,
    monkeypatch,
):
    client, _ = mock_mongo_api

    def should_not_be_called():
        raise AssertionError(
            "R2 must not be initialised"
        )

    monkeypatch.setattr(
        audio,
        "get_r2_storage",
        should_not_be_called,
    )

    response = post_audio(
        client,
        filename="sample.txt",
        content=b"not-audio",
        content_type="text/plain",
    )

    assert response.status_code == 400


def test_empty_audio_is_rejected_before_r2(
    mock_mongo_api,
    monkeypatch,
):
    client, _ = mock_mongo_api

    def should_not_be_called():
        raise AssertionError(
            "R2 must not be initialised"
        )

    monkeypatch.setattr(
        audio,
        "get_r2_storage",
        should_not_be_called,
    )

    response = post_audio(
        client,
        content=b"",
    )

    assert_error_response(
        response,
        400,
        "BAD_REQUEST",
        "Empty file",
    )


def test_oversized_audio_is_rejected_before_r2(
    mock_mongo_api,
    monkeypatch,
):
    client, _ = mock_mongo_api

    monkeypatch.setattr(
        audio,
        "_upload_size",
        lambda file: audio.MAX_UPLOAD_BYTES + 1,
    )

    def should_not_be_called():
        raise AssertionError(
            "R2 must not be initialised"
        )

    monkeypatch.setattr(
        audio,
        "get_r2_storage",
        should_not_be_called,
    )

    response = post_audio(client)

    assert_error_response(
        response,
        413,
        "PAYLOAD_TOO_LARGE",
        "File too large (max 30MB)",
    )