$(function () {
  const DASHBOARD_REFRESH_INTERVAL = 15000;
  const REQUEST_TIMEOUT = 10000;
  const DONATION_GOAL = 1000;

  const pageState = createAdminPageState();
  let refreshInProgress = false;

  // Live Event Log chart
  const eventChart = new ApexCharts(
    document.querySelector("#chart"),
    {
      series: [],

      chart: {
        type: "bar",
        height: 345,
        offsetX: -15,
        toolbar: {
          show: true
        },
        foreColor: "#adb0bb",
        fontFamily: "inherit"
      },

      colors: [
        "#5D87FF",
        "#49BEFF",
        "#13DEB9",
        "#FA896B"
      ],

      plotOptions: {
        bar: {
          horizontal: false,
          columnWidth: "35%",
          borderRadius: 6,
          borderRadiusApplication: "end"
        }
      },

      dataLabels: {
        enabled: false
      },

      legend: {
        show: true
      },

      grid: {
        borderColor: "rgba(0,0,0,0.1)",
        strokeDashArray: 3,
        xaxis: {
          lines: {
            show: false
          }
        }
      },

      xaxis: {
        type: "category",
        categories: []
      },

      yaxis: {
        show: true,
        min: 0,
        forceNiceScale: true
      },

      stroke: {
        show: true,
        width: 3,
        colors: ["transparent"]
      },

      noData: {
        text: "No detection data available"
      },

      tooltip: {
        theme: "light"
      }
    }
  );

  // Donation chart
  const donationChart = new ApexCharts(
    document.querySelector("#breakup"),
    {
      series: [],
      labels: [],

      chart: {
        width: 180,
        type: "donut",
        fontFamily: "Plus Jakarta Sans, sans-serif",
        foreColor: "#adb0bb"
      },

      plotOptions: {
        pie: {
          donut: {
            size: "75%"
          }
        }
      },

      stroke: {
        show: false
      },

      dataLabels: {
        enabled: false
      },

      legend: {
        show: false
      },

      colors: [
        "#5D87FF",
        "#49BEFF",
        "#13DEB9",
        "#FA896B",
        "#adb5bd"
      ],

      noData: {
        text: "No donation data"
      },

      responsive: [
        {
          breakpoint: 991,
          options: {
            chart: {
              width: 150
            }
          }
        }
      ],

      tooltip: {
        theme: "dark",
        fillSeriesColor: false,

        y: {
          formatter: value =>
            `$${Number(value).toFixed(2)}`
        }
      }
    }
  );

  // Map Visits chart
  // No active production API currently provides map visit data.
  const visitsChart = new ApexCharts(
    document.querySelector("#visited"),
    {
      chart: {
        id: "sparkline3",
        type: "area",
        height: 60,
        sparkline: {
          enabled: true
        },
        fontFamily: "Plus Jakarta Sans, sans-serif"
      },

      series: [],

      stroke: {
        curve: "smooth",
        width: 2
      },

      fill: {
        colors: ["#f3feff"],
        type: "solid",
        opacity: 0.05
      },

      markers: {
        size: 0
      },

      noData: {
        text: "No visit data available"
      },

      tooltip: {
        theme: "dark",
        x: {
          show: false
        }
      }
    }
  );

  async function fetchJson(url) {
    const controller = new AbortController();

    const timeoutId = setTimeout(
      () => controller.abort(),
      REQUEST_TIMEOUT
    );

    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          Accept: "application/json"
        }
      });

      if (!response.ok) {
        throw new Error(
          `${url} returned status ${response.status}`
        );
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  function donationBreakdown(donations) {
    const totals = new Map();

    donations
      .filter(
        item =>
          String(item.status).toLowerCase() ===
          "succeeded"
      )
      .forEach(item => {
        const type = item.type || "Donation";
        const amount = Number(item.amount || 0);

        totals.set(
          type,
          (totals.get(type) || 0) + amount
        );
      });

    return {
      labels: [...totals.keys()],
      values: [...totals.values()]
    };
  }

  async function loadDashboardData() {
    if (refreshInProgress) {
      return;
    }

    refreshInProgress = true;

    pageState.hideError();
    pageState.showLoading(
      "Loading dashboard data..."
    );

    try {
      const [
        detections,
        donationResponse
      ] = await Promise.all([
        fetchJson(
          "/detections/dashboard-summary?days=8"
        ),
        fetchJson("/donations")
      ]);

      // Update Live Event Log
      await eventChart.updateOptions({
        series: Array.isArray(detections.series)
          ? detections.series
          : [],

        xaxis: {
          categories: Array.isArray(
            detections.labels
          )
            ? detections.labels
            : []
        }
      });

      // Update Donation Progress
      const donations =
        donationResponse.charges?.data || [];

      const breakdown =
        donationBreakdown(donations);

      const total = breakdown.values.reduce(
        (sum, amount) => sum + amount,
        0
      );

      const totalElement =
        document.getElementById(
          "totalDonations"
        );

      if (total > 0) {
        totalElement.textContent =
          `Total Received: $${total.toFixed(2)} ` +
          `of $${DONATION_GOAL.toFixed(2)}`;
      } else {
        totalElement.textContent =
          "No successful donations available";
      }

      await donationChart.updateOptions({
        labels: breakdown.labels,
        series: breakdown.values
      });
    } catch (error) {
      console.error(
        "Dashboard refresh failed:",
        error
      );

      if (error.name === "AbortError") {
        pageState.showError(
          "Dashboard data timed out. Please try again."
        );
      } else {
        pageState.showError(
          "Dashboard data could not be loaded. Please try again."
        );
      }
    } finally {
      refreshInProgress = false;
      pageState.hideLoading();
    }
  }

  Promise.all([
    eventChart.render(),
    donationChart.render(),
    visitsChart.render()
  ])
    .then(loadDashboardData)
    .catch(error => {
      console.error(
        "Dashboard chart setup failed:",
        error
      );
    });

  window.retryDashboardData =
    loadDashboardData;

  setInterval(
    loadDashboardData,
    DASHBOARD_REFRESH_INTERVAL
  );
});