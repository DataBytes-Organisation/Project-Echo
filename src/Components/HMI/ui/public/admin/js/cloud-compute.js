// public/admin/js/cloud-compute.js

// Main entry point: runs when the page has loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Cloud Compute page JS loaded');
  
    setupControlPanelActions();
    initDummyMetrics();
    initDummyCharts();
  });
  
  //
  // 1. CONTROL PANEL BUTTONS
  //
  function setupControlPanelActions() {
    const btnScaleUp = document.getElementById('btn-scale-up');
    const btnScaleDown = document.getElementById('btn-scale-down');
    const btnCustomiseAlerts = document.getElementById('btn-customise-alerts');
  
    if (btnScaleUp) {
      btnScaleUp.addEventListener('click', () => {
        alert('Scale UP resources (dummy action)');
      });
    }
  
    if (btnScaleDown) {
      btnScaleDown.addEventListener('click', () => {
        alert('Scale DOWN resources (dummy action)');
      });
    }
  
    if (btnCustomiseAlerts) {
      btnCustomiseAlerts.addEventListener('click', () => {
        alert('Open "Customise Alerts" settings (dummy action)');
      });
    }
  }
  
  //
  // 2. DUMMY METRICS FOR THE TOP CARDS
  //
  function initDummyMetrics() {
    // Dummy values used for demonstration before API integration
    const dummyMetrics = {
      cpuUsage: 68,
      cpuTrend: 'Rising ↑',
      memoryUsage: '9 / 16 GB',
      memoryLoad: 'Avg load: High',
      storageUsage: '1.7 / 3 TB',
      storageGrowth: 'Growth: 10% weekly',
      storageForecast: 'Forecast: Full in ~2 months',
      currentCost: '$342.80 AUD',
      forecastCost: '$565.40 AUD'
    };
  
    // Update HTML elements by ID
    document.getElementById('cpu-usage-value').textContent = dummyMetrics.cpuUsage + '%';
    document.getElementById('cpu-trend-text').textContent = 'Trend: ' + dummyMetrics.cpuTrend;
  
    document.getElementById('memory-usage-value').textContent = dummyMetrics.memoryUsage;
    document.getElementById('memory-load-text').textContent = dummyMetrics.memoryLoad;
  
    document.getElementById('storage-usage-value').textContent = dummyMetrics.storageUsage;
    document.getElementById('storage-growth-text').textContent = dummyMetrics.storageGrowth;
    document.getElementById('storage-forecast-text').textContent = dummyMetrics.storageForecast;
  
    document.getElementById('current-cost-amount').textContent = dummyMetrics.currentCost;
    document.getElementById('forecast-cost-amount').textContent = dummyMetrics.forecastCost;
  }
  
  //
  // 3. DUMMY CHARTS USING APEXCHARTS
  //
  function initDummyCharts() {
    if (typeof ApexCharts === 'undefined') {
      console.warn('ApexCharts is not available');
      return;
    }
  
    // ----- Usage Chart (CPU / Memory / Storage) -----
    const usageChartElement = document.querySelector('#usage-chart');
    if (usageChartElement) {
      const usageOptions = {
        chart: {
          type: 'line',
          height: 250,
          toolbar: { show: false }
        },
        series: [
          { name: 'CPU', data: [40, 55, 50, 65, 60, 70] },
          { name: 'Memory', data: [60, 62, 64, 66, 67, 68] },
          { name: 'Storage', data: [30, 32, 34, 35, 36, 37] }
        ],
        xaxis: {
          categories: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        }
      };
  
      const usageChart = new ApexCharts(usageChartElement, usageOptions);
      usageChart.render();
    }
  
    // ----- Cost Chart -----
    const costChartElement = document.querySelector('#cost-chart');
    if (costChartElement) {
      const costOptions = {
        chart: {
          type: 'area',
          height: 220,
          toolbar: { show: false }
        },
        series: [
          {
            name: 'Cost (AUD)',
            data: [200, 230, 250, 280, 310, 340] // dummy cost trend
          }
        ],
        xaxis: {
          categories: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6']
        }
      };
  
      const costChart = new ApexCharts(costChartElement, costOptions);
      costChart.render();
    }
  }
  