document.addEventListener('DOMContentLoaded', function () {
  // Get references to HTML elements that we will interact with.
  const enquiryTableBody = document.getElementById('enquiry-table-body'); // The <tbody> of the table where we'll put the data.
  const enquiryModal = document.getElementById('enquiry-modal'); // The modal (popup) for displaying enquiry details.
  const enquiryDetailsContainer = document.getElementById('enquiry-details'); // The container inside the modal where details go.
  const closeButton = enquiryModal.querySelector('.close-button'); // The button to close the modal.
  const statusSelect = document.getElementById('status-select'); // The <select> dropdown for changing the status.
  const saveStatusButton = document.getElementById('save-status-button'); // The button to save the selected status.
  let currentEnquiryId = null; // Store the ID of the enquiry currently being viewed in the modal.
  let currentEnquiryData = null; // Store the data of the enquiry currently being viewed.

  // Function to open the enquiry details modal.
  function openEnquiryModal(enquiryId) {
    currentEnquiryId = enquiryId; // Store the ID.
    const row = document.querySelector(`[data-enquiry-id="${enquiryId}"]`); // Find the table row with the matching data-enquiry-id.
    if (row) {
      // Extract the data from the table row.  We use querySelector to get the specific elements
      // within the row's <td> cells.
      const username = row.querySelector('td:nth-child(1) h6').textContent;       // Username is in a <h6>
      const userDetail = row.querySelector('td:nth-child(2) span').textContent;    // User Detail is in a <span>
      const content = row.querySelector('td:nth-child(3) p').textContent;         // Content is in a <p>
      const category = row.querySelector('td:nth-child(4) span').textContent;    // Category is in a <span>
      const date = row.querySelector('td:nth-child(5) h6').textContent;          // Date is in a <h6>

      currentEnquiryData = {  // Store the data in an object.
        username: username,
        userDetail: userDetail,
        content: content,
        category: category,
        date: date,
        id: enquiryId
      };

      // Populate the modal's HTML with the enquiry details.
      enquiryDetailsContainer.innerHTML = `
        <p><strong>Username:</strong> ${currentEnquiryData.username}</p>
        <p><strong>User Detail:</strong> ${currentEnquiryData.userDetail}</p>
        <p><strong>Content:</strong> ${currentEnquiryData.content}</p>
        <p><strong>Category:</strong> ${currentEnquiryData.category}</p>
        <p><strong>Date:</strong> ${currentEnquiryData.date}</p>
        <p><strong>Enquiry ID:</strong> ${currentEnquiryData.id}</p>
        `;
      statusSelect.value = 'open'; // Reset the status dropdown.
      enquiryModal.style.display = 'block'; // Show the modal.
    } else {
      console.error(`Enquiry with ID ${enquiryId} not found in table.`); // Error message if the row isn't found.
    }
  }

  // Event listener for clicks within the table body.  This is used to detect clicks on the "Enquiry" buttons.
  enquiryTableBody.addEventListener('click', function (event) {
    if (event.target.classList.contains('view-enquiry-button')) { // Check if the clicked element has the class "view-enquiry-button".
      const enquiryId = event.target.dataset.enquiryId; // Get the enquiry ID from the data-enquiry-id attribute.
      openEnquiryModal(enquiryId); // Call the function to open the modal.
    }
  });

  // Event listener for the modal's close button.
  closeButton.addEventListener('click', function () {
    enquiryModal.style.display = 'none'; // Hide the modal.
    currentEnquiryId = null;             // Clear the stored ID and data.
    currentEnquiryData = null;
  });

  // Event listener for clicks outside the modal.  This also closes the modal.
  window.addEventListener('click', function (event) {
    if (event.target === enquiryModal) { // Check if the click target is the modal itself (the background area).
      enquiryModal.style.display = 'none'; // Hide the modal.
      currentEnquiryId = null;
      currentEnquiryData = null;
    }
  });

  // Event listener for the "Save Status" button.
  saveStatusButton.addEventListener('click', function () {
    if (currentEnquiryId && currentEnquiryData) { // Make sure we have an enquiry ID.
      const newStatus = statusSelect.value;       // Get the selected status from the dropdown.
      console.log(`Updating enquiry ${currentEnquiryId} to status: ${newStatus}`); // Log the update.
      updateEnquiryStatus(currentEnquiryId, newStatus); // Call the function to update the status.
      enquiryModal.style.display = 'none';         // Hide the modal.
      currentEnquiryId = null;
      currentEnquiryData = null;
    }
  });

  // Function to simulate updating the enquiry status.
  function updateEnquiryStatus(id, status) {
    console.log(`Simulating API call to update enquiry ${id} to status: ${status}`);
    //  Replace this with your actual API call to update the status on the server.
  }

  // Function to load and display the enquiries.
  function loadEnquiries() {
    // Sample enquiry data.  In a real application, this would come from your server.
    const sampleEnquiries = [
      { id: '1', username: 'User One', userDetail: 'Detail 1', content: 'This is enquiry content one.', category: 'Request', date: '2025-04-28' },
      { id: '2', username: 'User Two', userDetail: 'Detail 2', content: 'Enquiry number two here.', category: 'Enquiry', date: '2025-04-27' },
      { id: '3', username: 'User Three', userDetail: 'Detail 3', content: 'A third enquiry to display.', category: 'Request', date: '2025-04-26' },
    ];

    // Simulate an API call using Promise.resolve().  This makes the code more like a real-world scenario
    // where you'd fetch data from a server using fetch().
    Promise.resolve({ json: () => Promise.resolve(sampleEnquiries) })
      .then(response => response.json()) // Parse the JSON from the (simulated) response.
      .then(data => {
        const tbody = document.getElementById('enquiry-table-body'); // Get the table body.
        tbody.innerHTML = ''; // Clear the table body before adding new rows.  This is important!

        // Iterate over the enquiry data and create a table row for each enquiry.
        data.forEach(entry => {
          const row = document.createElement('tr'); // Create a new table row (<tr>).
          // Set a data attribute on the row to store the enquiry ID.  This is used to find the row when the "Enquiry" button is clicked.
          row.dataset.enquiryId = entry.id || entry._id || entry.username.replace(/\s+/g, '-').toLowerCase() + '-' + new Date().getTime();
          row.innerHTML = `
            <td class="border-bottom-0"><h6 class="fw-semibold mb-0">${entry.username}</h6></td>  
            <td class="border-bottom-0">
              <h6 class="fw-semibold mb-1">${entry.userDetail}</h6>
              <span class="fw-normal">${entry.userDetail}</span>
            </td>
            <td class="border-bottom-0">
              <p class="mb-0 fw-normal">${entry.content}</p>
            </td>
            <td class="border-bottom-0">
              <div class="d-flex align-items-center gap-2">
                <span class="badge ${entry.category === 'Request' ? 'bg-secondary' : 'bg-success'} rounded-3 fw-semibold">
                  ${entry.category}
                </span>
              </div>
            </td>
            <td class="border-bottom-0">
              <h6 class="fw-semibold mb-0 fs-4">${entry.date}</h6>
            </td>
            <td class="border-bottom-0">
              <button class="btn btn-sm btn-primary view-enquiry-button" data-enquiry-id="${row.dataset.enquiryId}">Enquiry</button>
            </td>
          `;
          tbody.appendChild(row); // Add the new row to the table body.
        });
      })
      .catch(error => {
        console.error("Failed to load enquiries:", error); // Handle errors during the fetch.
      });
  }

  // Call the function to load the enquiries when the page has finished loading.
  loadEnquiries();
});