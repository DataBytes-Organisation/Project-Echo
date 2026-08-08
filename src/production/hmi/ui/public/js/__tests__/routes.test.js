/**
 * @jest-environment jsdom
 *
 * withRetry (imported from HMI-utils.js) calls showToast internally between
 * attempts, and that reaches straight for document/document.body - turns out you
 * can't just mock showToast out from outside since it's a local call within that
 * module, not something routed through the exports object. Easiest fix is just
 * giving it a real DOM to work with instead of fighting the mock.
 */

// under jsdom, routes.js goes down its "window.axios" branch (same as in the
// real browser, where axios gets loaded via a CDN <script> tag onto window
// before routes.js runs) - so we set that up directly rather than mocking
// require('axios'), which wouldn't be the path actually taken here
const mockApi = { get: jest.fn(), post: jest.fn(), patch: jest.fn() };
global.window.axios = { create: jest.fn(() => mockApi) };

const routes = require('../routes.js');

beforeEach(() => {
  mockApi.get.mockReset();
  mockApi.post.mockReset();
  mockApi.patch.mockReset();
});

describe('routes.js - GET retry behaviour', () => {
  it('retries a failing GET up to 3 attempts and succeeds if a later attempt works', async () => {
    mockApi.get
      .mockRejectedValueOnce(new Error('blip 1'))
      .mockRejectedValueOnce(new Error('blip 2'))
      .mockResolvedValueOnce({ data: [{ _id: 'node-1' }] });

    const result = await routes.retrieveIotNodes();

    expect(mockApi.get).toHaveBeenCalledTimes(3);
    expect(result.data).toEqual([{ _id: 'node-1' }]);
  }, 10000);

  it('gives up after 3 attempts and throws if it never succeeds', async () => {
    mockApi.get.mockRejectedValue(new Error('permanently down'));

    await expect(routes.retrieveMicrophones()).rejects.toThrow('permanently down');
    expect(mockApi.get).toHaveBeenCalledTimes(3);
  }, 10000);

  it('does not retry at all on the first-try success (no wasted calls)', async () => {
    mockApi.get.mockResolvedValueOnce({ data: [] });

    await routes.retrieveMicrophones();

    expect(mockApi.get).toHaveBeenCalledTimes(1);
  });

  it('retrieveIotNode hits the relative proxy path, not the API host directly', async () => {
    mockApi.get.mockResolvedValueOnce({ data: { _id: 'node_1' } });

    const result = await routes.retrieveIotNode('node_1');

    expect(mockApi.get).toHaveBeenCalledWith('/iot/nodes/node_1');
    expect(result.data).toEqual({ _id: 'node_1' });
  });

  it('retrieveIotNode encodes ids that would otherwise break the path', async () => {
    mockApi.get.mockResolvedValueOnce({ data: {} });

    await routes.retrieveIotNode('node/with space');

    expect(mockApi.get).toHaveBeenCalledWith('/iot/nodes/node%2Fwith%20space');
  });
});

describe('routes.js - state-changing requests are NOT retried', () => {
  it('postRecording only tries once, even on failure', async () => {
    mockApi.post.mockRejectedValueOnce(new Error('backend rejected it'));

    await expect(routes.postRecording({ foo: 'bar' })).rejects.toThrow('backend rejected it');
    expect(mockApi.post).toHaveBeenCalledTimes(1);
  });

  it('setSimModeAnimal (sim control POST) only tries once', async () => {
    mockApi.post.mockRejectedValueOnce(new Error('sim unavailable'));

    await expect(routes.setSimModeAnimal()).rejects.toThrow('sim unavailable');
    expect(mockApi.post).toHaveBeenCalledTimes(1);
  });

  it('updateRequestStatus (PATCH) only tries once', async () => {
    mockApi.patch.mockRejectedValueOnce(new Error('conflict'));

    await expect(routes.updateRequestStatus('req-1', 'approved')).rejects.toThrow('conflict');
    expect(mockApi.patch).toHaveBeenCalledTimes(1);
  });

  it('updateConservationStatus (PATCH) only tries once', async () => {
    mockApi.patch.mockRejectedValueOnce(new Error('conflict'));

    await expect(routes.updateConservationStatus('koala', 'endangered')).rejects.toThrow('conflict');
    expect(mockApi.patch).toHaveBeenCalledTimes(1);
  });

  it('signIn (POST) only tries once - credentials must never be replayed', async () => {
    mockApi.post.mockRejectedValueOnce(new Error('invalid credentials'));

    await expect(
      routes.signIn({ username: 'someone', email: '', password: 'wrong' })
    ).rejects.toThrow('invalid credentials');
    expect(mockApi.post).toHaveBeenCalledTimes(1);
  });

  it('signIn posts to the Node auth route with the credentials it was given', async () => {
    mockApi.post.mockResolvedValueOnce({ data: { token: 't', userId: 'u1' } });

    await routes.signIn({ username: 'someone', email: 'a@b.c', password: 'pw' });

    expect(mockApi.post).toHaveBeenCalledWith('/api/auth/signin', {
      username: 'someone',
      email: 'a@b.c',
      password: 'pw',
    });
  });
});

describe('routes.js - background callers can silence the retry toasts', () => {
  // toasts linger in the shared jsdom document between tests, so compare deltas
  const countToasts = () => document.querySelectorAll('.hmi-toast').length;

  it('retrieveIotNodes passes silent through to withRetry but keeps retrying', async () => {
    const before = countToasts();
    mockApi.get
      .mockRejectedValueOnce(new Error('blip'))
      .mockResolvedValueOnce({ data: [] });

    await routes.retrieveIotNodes({ silent: true });

    expect(mockApi.get).toHaveBeenCalledTimes(2);
    expect(countToasts() - before).toBe(0);
  }, 10000);

  it('retrieveSensorAlerts still toasts when not asked to be silent', async () => {
    const before = countToasts();
    mockApi.get
      .mockRejectedValueOnce(new Error('blip'))
      .mockResolvedValueOnce({ data: { items: [] } });

    await routes.retrieveSensorAlerts();

    expect(mockApi.get).toHaveBeenCalledTimes(2);
    expect(countToasts() - before).toBe(1);
  }, 10000);
});
