document.getElementById('registerForm').addEventListener('submit', function (e) {
    e.preventDefault(); // Prevent the default form submission

    // Get form data
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm_password').value;
    const role = document.getElementById('role').value;
    

    // Validate passwords
    if (password !== confirmPassword) {
        alert("Passwords do not match!");
        return;
    }

    // Validate email format
    const emailPattern = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/;
    if (!emailPattern.test(email)) {
        alert("Invalid email format!");
        return;
    }

    // Prepare data to be sent to the server
    const data = {
        username: username,
        email: email,
        password: password,
        confirm_password: confirmPassword,
        role: role
    };

    // Send the data to the Flask backend using fetch API (POST request)
    fetch('/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error); // Show error messages if any
        } else {
            window.location.href = '/login'; // Redirect to login page on success
        }
    })
    .catch(error => console.error('Error:', error));
});
