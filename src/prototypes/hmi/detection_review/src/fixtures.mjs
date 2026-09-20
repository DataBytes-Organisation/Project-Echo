function freezeFixture(fixture) {
  return Object.freeze({
    ...fixture,
    microphoneLLA: Object.freeze([...fixture.microphoneLLA]),
    animalEstLLA: Object.freeze([...fixture.animalEstLLA]),
  });
}

export const detectionFixtures = Object.freeze([
  freezeFixture({
    id: "det-echo-001",
    timestamp: "2026-05-22T04:12:00.000Z",
    sensorId: "ECHO-07",
    species: "Powerful Owl",
    confidence: 94.6,
    microphoneLLA: [-37.8142, 144.9628, 18],
    animalEstLLA: [-37.8136, 144.9631, 24],
    animalLLAUncertainty: 18,
    audioAvailable: true,
    sampleRate: 48000,
    queueStatus: "pending_review",
    version: 1,
  }),
  freezeFixture({
    id: "det-echo-002",
    timestamp: "2026-05-22T03:48:30.000Z",
    sensorId: "ECHO-12",
    species: "Yellow-tailed Black-Cockatoo",
    confidence: 87.2,
    microphoneLLA: [-37.7321, 145.0247, 96],
    animalEstLLA: [-37.7314, 145.0255, 104],
    animalLLAUncertainty: 31,
    audioAvailable: false,
    sampleRate: 44100,
    queueStatus: "pending_review",
    version: 1,
  }),
  freezeFixture({
    id: "det-echo-003",
    timestamp: "2026-05-22T02:19:45.000Z",
    sensorId: "ECHO-04",
    species: "Eastern Whipbird",
    confidence: 78.9,
    microphoneLLA: [-38.1124, 145.3328, 42],
    animalEstLLA: [-38.1118, 145.3335, 47],
    animalLLAUncertainty: 44,
    audioAvailable: true,
    sampleRate: 48000,
    queueStatus: "pending_review",
    version: 1,
  }),
]);

export const malformedDetectionFixtures = Object.freeze([
  freezeFixture({
    id: "det-echo-invalid",
    timestamp: "2026-05-22T01:05:00.000Z",
    sensorId: "ECHO-99",
    species: "Unknown fixture",
    confidence: 101,
    microphoneLLA: [-37.8, 144.9, 16],
    animalEstLLA: [-37.79, 144.91, 20],
    animalLLAUncertainty: 25,
    audioAvailable: false,
    sampleRate: 48000,
    queueStatus: "pending_review",
    version: 1,
  }),
]);
