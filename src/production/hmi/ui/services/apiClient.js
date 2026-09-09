const axios = require('axios');

const API_HOST = process.env.API_HOST || 'localhost';
const API_BASE_URL = `http://${API_HOST}:9000`;
const DEFAULT_TIMEOUT_MS = 10000;

// Throwing this for every failure case so callers can just check err.isNetworkError / err.status
// instead of digging through axios's err.response / err.request shape every single time.
class ApiError extends Error {
  constructor(message, { status = null, isNetworkError = false, data = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.isNetworkError = isNetworkError;
    this.data = data;
  }
}

async function apiRequest(method, path, { params, data, headers, baseURL, timeout } = {}) {
  try {
    const response = await axios({
      method,
      url: `${baseURL || API_BASE_URL}${path}`,
      params,
      data,
      headers,
      timeout: timeout || DEFAULT_TIMEOUT_MS,
    });
    return response.data;
  } catch (err) {
    if (err.response) {
      // Backend actually got the request but came back with a non-2xx status
      throw new ApiError(
        `Backend responded ${err.response.status} for ${method.toUpperCase()} ${path}`,
        { status: err.response.status, data: err.response.data }
      );
    }
    if (err.request) {
      // Couldn't reach the backend at all - could be down, timed out, DNS being weird, etc.
      throw new ApiError(
        `No response from backend for ${method.toUpperCase()} ${path}: ${err.message}`,
        { isNetworkError: true }
      );
    }
    // Didn't even get sent - probably a bad config somewhere
    throw new ApiError(`Request setup failed for ${method.toUpperCase()} ${path}: ${err.message}`);
  }
}

// Turns an ApiError (or whatever else gets thrown) into a consistent JSON error response
function sendApiError(res, err, fallbackMessage = 'Something went wrong') {
  if (err instanceof ApiError) {
    if (err.isNetworkError) {
      console.error(err.message);
      return res.status(502).json({ error: 'API unavailable' });
    }
    console.error(err.message, err.data);
    return res.status(err.status || 500).json({ error: err.data?.message || fallbackMessage });
  }
  console.error(err);
  return res.status(500).json({ error: fallbackMessage });
}

module.exports = {
  API_BASE_URL,
  ApiError,
  sendApiError,
  get: (path, opts) => apiRequest('get', path, opts),
  post: (path, data, opts = {}) => apiRequest('post', path, { ...opts, data }),
  patch: (path, data, opts = {}) => apiRequest('patch', path, { ...opts, data }),
  delete: (path, opts) => apiRequest('delete', path, opts),
};
