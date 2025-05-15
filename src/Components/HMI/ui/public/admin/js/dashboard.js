$(function () {
  // =====================================
  // Profit
  // =====================================
  var chart = {
    series: [
      { name: "Microphone 1", data: [355, 390, 300, 350, 390, 180, 355, 390] },
      { name: "Microphone 2", data: [280, 250, 325, 215, 250, 310, 280, 250] },
    ],
    chart: {
      type: "bar",
      height: 345,
      offsetX: -15,
      toolbar: { show: true },
      foreColor: "#adb0bb",
      fontFamily: 'inherit',
      sparkline: { enabled: false },
    },
    colors: ["#5D87FF", "#49BEFF"],
    plotOptions: {
      bar: {
        horizontal: false,
        columnWidth: "35%",
        borderRadius: [6],
        borderRadiusApplication: 'end',
        borderRadiusWhenStacked: 'all'
      },
    },
    markers: { size: 0 },
    dataLabels: { enabled: false },
    legend: { show: false },
    grid: {
      borderColor: "rgba(0,0,0,0.1)",
      strokeDashArray: 3,
      xaxis: { lines: { show: false } },
    },
    xaxis: {
      type: "category",
      categories: ["16/08", "17/08", "18/08", "19/08", "20/08", "21/08", "22/08", "23/08"],
      labels: { style: { cssClass: "grey--text lighten-2--text fill-color" } },
    },
    yaxis: {
      show: true,
      min: 0,
      max: 400,
      tickAmount: 4,
      labels: { style: { cssClass: "grey--text lighten-2--text fill-color" } },
    },
    stroke: {
      show: true,
      width: 3,
      lineCap: "butt",
      colors: ["transparent"],
    },
    tooltip: { theme: "light" },
    responsive: [{
      breakpoint: 600,
      options: {
        plotOptions: {
          bar: { borderRadius: 3 }
        }
      }
    }]
  };
  new ApexCharts(document.querySelector("#chart"), chart).render();

  // =====================================
  // Percentage - Donation
  // =====================================
  const series = [50, 438];
  const sum = series.reduce((a, b) => a + b, 0);
  const breakup = {
    color: "#adb5bd",
    endgoal: 1000,
    series: series,
    labels: ["Donation", "Subscription"],
    chart: {
      width: 180,
      type: "donut",
      fontFamily: "Plus Jakarta Sans', sans-serif",
      foreColor: "#adb0bb",
    },
    plotOptions: {
      pie: {
        startAngle: 0,
        endAngle: Math.min(Math.round((sum / 1000) * 360), 360),
        donut: { size: '75%' },
      },
    },
    stroke: { show: false },
    dataLabels: { enabled: false },
    legend: { show: false },
    colors: ["#5D87FF", "#ecf2ff", "#F9F9FD"],
    responsive: [{
      breakpoint: 991,
      options: { chart: { width: 150 } }
    }],
    tooltip: { theme: "dark", fillSeriesColor: false },
  };
  new ApexCharts(document.querySelector("#breakup"), breakup).render();

  // =====================================
  // Earning
  // =====================================
  var visited = {
    chart: {
      id: "sparkline3",
      type: "area",
      height: 60,
      sparkline: { enabled: true },
      group: "sparklines",
      fontFamily: "Plus Jakarta Sans', sans-serif",
      foreColor: "#adb0bb",
    },
    series: [{ name: "visited", color: "#49BEFF", data: [25, 66, 20, 40, 12, 58, 20] }],
    stroke: { curve: "smooth", width: 2 },
    fill: {
      colors: ["#f3feff"],
      type: "solid",
      opacity: 0.05,
    },
    markers: { size: 0 },
    tooltip: {
      theme: "dark",
      fixed: { enabled: true, position: "right" },
      x: { show: false },
    },
  };
  new ApexCharts(document.querySelector("#visited"), visited).render();
});

// =====================================
// Fetch and render enquiries
// =====================================
fetch("/api/enquiries")
  .then(res => res.json())
  .then(data => {
    const tbody = document.getElementById("enquiry-table-body");
    tbody.innerHTML = ""; // Clear existing rows

    data.forEach(e => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td class="border-bottom-0"><h6 class="fw-semibold mb-0">${e.username}</h6></td>
        <td class="border-bottom-0"><h6 class="fw-semibold mb-1">${e.userDetail}</h6></td>
        <td class="border-bottom-0"><p class="mb-0 fw-normal">${e.content}</p></td>
        <td class="border-bottom-0"><span class="badge ${e.category === 'Request' ? 'bg-secondary' : 'bg-success'}">${e.category}</span></td>
        <td class="border-bottom-0"><h6 class="fw-semibold mb-0 fs-4">${e.date}</h6></td>
      `;
      row.dataset.id = e._id;
      row.addEventListener("click", () => openModalWithEnquiry(e));
      tbody.appendChild(row);
    });
  })
  .catch(err => {
    console.error("Failed to load enquiries:", err);
  });

// =====================================
// Open modal and populate with enquiry
// =====================================
function openModalWithEnquiry(enquiry) {
  document.getElementById("modal-user-name").textContent = enquiry.username;
  document.getElementById("modal-user-role").textContent = enquiry.userDetail;
  document.getElementById("modal-enquiry-status").value = enquiry.status || "Open";
  document.getElementById("modal-enquiry-notes").value = enquiry.notes || "";

  document.getElementById("saveEnquiryButton").dataset.id = enquiry._id;

  const modal = new bootstrap.Modal(document.getElementById("userModal"));
  modal.show();
}

// =====================================
// Save updated enquiry
// =====================================
document.getElementById("saveEnquiryButton").addEventListener("click", () => {
  const id = document.getElementById("saveEnquiryButton").dataset.id;
  const status = document.getElementById("modal-enquiry-status").value;
  const notes = document.getElementById("modal-enquiry-notes").value;

  fetch(`/api/enquiries/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, notes }),
  })
    .then(res => {
      if (!res.ok) throw new Error("Failed to update enquiry");
      return res.json();
    })
    .then(() => {
      alert("Enquiry updated successfully.");
      location.reload();
    })
    .catch(err => {
      console.error(err);
      alert("Error updating enquiry.");
    });
});
