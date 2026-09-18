if (typeof jest === 'undefined') {
  const assert = require('node:assert/strict');
  const test = require('node:test');

  const queue = [];
  function axiosMock() {
    const next = queue.shift();
    if (next?.reject) return Promise.reject(next.reject);
    return Promise.resolve(next?.resolve);
  }
  axiosMock.mockResolvedValueOnce = (value) => queue.push({ resolve: value });
  axiosMock.mockRejectedValueOnce = (value) => queue.push({ reject: value });

  const axiosPath = require.resolve('axios');
  require.cache[axiosPath] = {
    id: axiosPath,
    filename: axiosPath,
    loaded: true,
    exports: axiosMock,
  };

  const { ApiError, sendApiError, get, post } = require('../../../server/services/apiClient');

  function fakeRes() {
    const res = {};
    res.statusCode = null;
    res.jsonBody = null;
    res.status = (code) => { res.statusCode = code; return res; };
    res.json = (body) => { res.jsonBody = body; return res; };
    return res;
  }

  test('apiClient returns response.data on successful GET and POST', async () => {
    axiosMock.mockResolvedValueOnce({ data: { hello: 'world' } });
    assert.deepEqual(await get('/some/path'), { hello: 'world' });

    axiosMock.mockResolvedValueOnce({ data: { created: true } });
    assert.deepEqual(await post('/some/path', { name: 'test' }), { created: true });
  });

  test('apiClient wraps backend 4xx/5xx errors as ApiError', async () => {
    axiosMock.mockRejectedValueOnce({
      response: { status: 404, data: { detail: 'not found' } },
    });
    await assert.rejects(get('/missing'), {
      status: 404,
      isNetworkError: false,
      data: { detail: 'not found' },
    });

    axiosMock.mockRejectedValueOnce({
      response: { status: 500, data: { detail: 'server exploded' } },
    });
    await assert.rejects(get('/broken'), ApiError);
  });

  test('apiClient distinguishes network and setup failures', async () => {
    axiosMock.mockRejectedValueOnce({ request: {}, message: 'timeout of 10000ms exceeded' });
    await assert.rejects(get('/unreachable'), {
      isNetworkError: true,
      status: null,
    });

    axiosMock.mockRejectedValueOnce({ message: 'Invalid URL' });
    await assert.rejects(get('/bad'), ApiError);
  });

  test('sendApiError returns safe user-facing messages', () => {
    const network = new ApiError('No response from backend for GET /x: connect ECONNREFUSED', {
      isNetworkError: true,
    });
    const networkRes = fakeRes();
    sendApiError(networkRes, network);
    assert.equal(networkRes.statusCode, 502);
    assert.deepEqual(networkRes.jsonBody, { error: 'API unavailable' });

    const backend = new ApiError('Backend responded 404 for GET /x', {
      status: 404,
      data: { message: 'Requested item not found' },
    });
    const backendRes = fakeRes();
    sendApiError(backendRes, backend);
    assert.equal(backendRes.statusCode, 404);
    assert.deepEqual(backendRes.jsonBody, { error: 'Requested item not found' });

    const fallbackRes = fakeRes();
    sendApiError(fallbackRes, new ApiError('Backend responded 400 for POST /x', { status: 400, data: null }), 'Custom fallback message');
    assert.equal(fallbackRes.statusCode, 400);
    assert.deepEqual(fallbackRes.jsonBody, { error: 'Custom fallback message' });

    const plainRes = fakeRes();
    sendApiError(plainRes, new Error('some unrelated bug'));
    assert.equal(plainRes.statusCode, 500);
    assert.deepEqual(plainRes.jsonBody, { error: 'Something went wrong' });
  });
} else {
  jest.mock('axios');

  const axios = require('axios');
  const { ApiError, sendApiError, get, post } = require('../../../server/services/apiClient');

  function fakeRes() {
    const res = {};
    res.statusCode = null;
    res.jsonBody = null;
    res.status = jest.fn((code) => { res.statusCode = code; return res; });
    res.json = jest.fn((body) => { res.jsonBody = body; return res; });
    return res;
  }

  describe('apiClient - response parsing', () => {
    it('returns response.data on a successful GET', async () => {
      axios.mockResolvedValueOnce({ data: { hello: 'world' } });
      const data = await get('/some/path');
      expect(data).toEqual({ hello: 'world' });
    });

    it('returns response.data on a successful POST', async () => {
      axios.mockResolvedValueOnce({ data: { created: true } });
      const data = await post('/some/path', { name: 'test' });
      expect(data).toEqual({ created: true });
    });
  });

  describe('apiClient - backend 4xx/5xx normalisation', () => {
    it('wraps a 404 as an ApiError with the real status and body', async () => {
      axios.mockRejectedValueOnce({
        response: { status: 404, data: { detail: 'not found' } },
      });

      await expect(get('/missing')).rejects.toMatchObject({
        status: 404,
        isNetworkError: false,
        data: { detail: 'not found' },
      });
    });

    it('wraps a 500 the same way', async () => {
      axios.mockRejectedValueOnce({
        response: { status: 500, data: { detail: 'server exploded' } },
      });

      await expect(get('/broken')).rejects.toBeInstanceOf(ApiError);
      axios.mockRejectedValueOnce({
        response: { status: 500, data: { detail: 'server exploded' } },
      });
      await expect(get('/broken')).rejects.toMatchObject({ status: 500 });
    });
  });

  describe('apiClient - network/timeout handling', () => {
    it('flags isNetworkError when the backend never responded at all', async () => {
      axios.mockRejectedValueOnce({ request: {}, message: 'timeout of 10000ms exceeded' });

      await expect(get('/unreachable')).rejects.toMatchObject({
        isNetworkError: true,
        status: null,
      });
    });

    it('still throws an ApiError even if the request never got sent (bad config)', async () => {
      axios.mockRejectedValueOnce({ message: 'Invalid URL' });

      await expect(get('/bad')).rejects.toBeInstanceOf(ApiError);
    });
  });

  describe('sendApiError - safe user-facing messages', () => {
    it('returns 502 for a network error, without leaking the raw error message', () => {
      const err = new ApiError('No response from backend for GET /x: connect ECONNREFUSED', {
        isNetworkError: true,
      });
      const res = fakeRes();

      sendApiError(res, err);

      expect(res.statusCode).toBe(502);
      expect(res.jsonBody).toEqual({ error: 'API unavailable' });
    });

    it('passes through the backend status code and a safe message for a normal ApiError', () => {
      const err = new ApiError('Backend responded 404 for GET /x', {
        status: 404,
        data: { message: 'Requested item not found' },
      });
      const res = fakeRes();

      sendApiError(res, err);

      expect(res.statusCode).toBe(404);
      expect(res.jsonBody).toEqual({ error: 'Requested item not found' });
    });

    it('falls back to the generic fallback message when the backend gave no message', () => {
      const err = new ApiError('Backend responded 400 for POST /x', { status: 400, data: null });
      const res = fakeRes();

      sendApiError(res, err, 'Custom fallback message');

      expect(res.statusCode).toBe(400);
      expect(res.jsonBody).toEqual({ error: 'Custom fallback message' });
    });

    it('treats anything that is not an ApiError as a plain 500, not a crash', () => {
      const res = fakeRes();

      sendApiError(res, new Error('some unrelated bug'));

      expect(res.statusCode).toBe(500);
      expect(res.jsonBody).toEqual({ error: 'Something went wrong' });
    });
  });
}
