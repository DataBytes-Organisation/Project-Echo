document.addEventListener("DOMContentLoaded", function () {
  const urlParams = new URLSearchParams(window.location.search);
  const selectedAnimalName = decodeURIComponent(urlParams.get("animal"));
  const percentage = parseFloat(urlParams.get("percentage"));
  const description = decodeURIComponent(urlParams.get("description"));

  const textBox = document.querySelector(".text-box p");
  const animalList = document.getElementById("animal-list");
  const chartCanvas = document.getElementById("myChart");
  let chartData = [];
  const chartColors = [
    "#3366CC",
    "#DC3912",
    "#FF9900",
    "#109618",
    "#990099",
    "#3B3EAC",
    "#0099C6",
    "#DD4477",
    "#66AA00",
    "#B82E2E",
    "#316395",
    "#994499",
    "#22AA99",
    "#AAAA11",
    "#6633CC",
    "#E67300",
    "#8B0707",
    "#651067",
    "#329262",
    "#5574A6",
    "#3B3EAC",
  ];

  textBox.innerHTML = selectedAnimalName
    ? `<strong>${selectedAnimalName}:</strong> ${description}`
    : "";

  animals.forEach((animal, index) => {
    const listItem = document.createElement("li");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `animal-${index}`;
    checkbox.value = animal.name;
    checkbox.checked = animal.name === selectedAnimalName;

    if (checkbox.checked) {
      const animalPercentage = ((animal.count / totalAnimals) * 100).toFixed(2);
      chartData.push({
        label: `${animal.name} ${animalPercentage}%`,
        data: animalPercentage,
      });
    }

    const label = document.createElement("label");
    label.htmlFor = `animal-${index}`;
    label.textContent = animal.name;

    listItem.appendChild(checkbox);
    listItem.appendChild(label);
    animalList.appendChild(listItem);

    listItem.addEventListener("click", (event) => {
      if (event.target !== checkbox) {
        checkbox.checked = !checkbox.checked;
      }
      checkbox.dispatchEvent(new Event("change"));
    });

    checkbox.addEventListener("change", () => {
      updateChartAndDescriptions(checkbox, animal);
    });
  });

  function updateChartAndDescriptions(checkbox, animal) {
    const animalPercentage = ((animal.count / totalAnimals) * 100).toFixed(2);
    if (checkbox.checked) {
      if (
        !chartData.some(
          (item) => item.label === `${animal.name} ${animalPercentage}%`
        )
      ) {
        chartData.push({
          label: `${animal.name} ${animalPercentage}%`,
          data: animalPercentage,
        });
        textBox.innerHTML += `<br><strong>${animal.name}:</strong> ${animal.description}`;
      }
    } else {
      chartData = chartData.filter(
        (item) => item.label !== `${animal.name} ${animalPercentage}%`
      );
      textBox.innerHTML = textBox.innerHTML.replace(
        `<br><strong>${animal.name}:</strong> ${animal.description}`,
        ""
      );
    }
    updateChart();
  }

  function updateChart() {
    if (chartData.length > 0) {
      const data = {
        datasets: [
          {
            data: chartData.map((item) => parseFloat(item.data)),
            backgroundColor: chartColors.slice(0, chartData.length),
          },
        ],
        labels: chartData.map((item) => item.label),
      };

      const config = {
        type: "pie",
        data,
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "right",
              labels: {
                color: "white",
              },
            },
          },
        },
      };

      if (window.myPieChart) {
        window.myPieChart.destroy();
      }
      window.myPieChart = new Chart(chartCanvas, config);
    } else {
      if (window.myPieChart) {
        window.myPieChart.destroy();
      }
      // Clear descriptions if no animals are selected
      textBox.innerHTML = "";
    }
  }

  // Manually trigger the initial chart update if there's an initially selected animal
  updateChart();
});
