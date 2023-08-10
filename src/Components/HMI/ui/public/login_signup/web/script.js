//Switching between username and email
function toggleLoginType() {
  const emailInput = document.getElementById('emailInput');
  const usernameInput = document.getElementById('usernameInput');
  const switchInput = document.getElementById('switchInput');

  if (switchInput.checked) {
    emailInput.style.display = 'none';
    usernameInput.style.display = 'block';

    document.getElementById('username').setAttribute('required', 'required');
    document.getElementById('email').removeAttribute('required');
  } else {
    emailInput.style.display = 'block';
    usernameInput.style.display = 'none';

    document.getElementById('username').setAttribute('required', 'required');
    document.getElementById('email').removeAttribute('required');
  }
}

//For Authentication, matching credentials from client side with those of database

document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  let identifier;
  if (document.getElementById('switchInput').checked) {
    identifier = document.getElementById('username').value;
  } else {
    identifier = document.getElementById('email').value;
  }
  const password = document.getElementById('password').value;
  const response = await fetch('/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ identifier, password })
  });

  //For redirecting user to welcome page, if the credentials are correct and a response.ok is received

  try {
    const data = await response.json();
    if (response.ok) {
      if (data.redirectTo) {
        window.open(data.redirectTo, "_blank");
      } else {
        alert(data.message);
      }
    } else {
      alert(data.error);
    }
  } catch (error) {
    console.error(error);
    alert('An error occurred.');
  }
});
