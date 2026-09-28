function registerRoutes(app, dependencies) {
  const { axios, apiClient, apiBaseUrl, checkUserSession } = dependencies;

  async function proxyToApi(req, res) {
    try {
      const url = `${apiBaseUrl}${req.originalUrl}`;
      const response = await axios({
        method: req.method,
        url,
        params: req.query,
        data: req.body,
        validateStatus: () => true,
      });

      if (res.headersSent) return;

      res.status(response.status);
      if (typeof response.data === 'undefined') return res.end();
      return res.send(response.data);
    } catch (error) {
      console.error('Error proxying to API:', error.message);
      if (!res.headersSent) {
        return res.status(502).json({ error: 'API unavailable' });
      }
    }
  }

  app.all('/sensors', proxyToApi);
  app.all('/sensors/*', proxyToApi);
  app.all('/mqtt', proxyToApi);
  app.all('/mqtt/*', proxyToApi);
  app.all('/hmi/*', proxyToApi);

  app.get('/iot/nodes', checkUserSession, async (req, res) => {
    try {
      const response = await axios.get(`${apiBaseUrl}/iot/nodes`, {
        headers: { Authorization: `Bearer ${req.session.token}` },
        timeout: 10000,
      });
      res.json(response.data);
    } catch (error) {
      apiClient.sendApiError(res, error, 'Error fetching IoT nodes');
    }
  });

  app.get('/iot/nodes/:nodeId', async (req, res) => {
    try {
      const data = await apiClient.get(`/iot/nodes/${encodeURIComponent(req.params.nodeId)}`);
      res.json(data);
    } catch (error) {
      apiClient.sendApiError(res, error, 'Error fetching IoT node details');
    }
  });
}

module.exports = { registerRoutes };
