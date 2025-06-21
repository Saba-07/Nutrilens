function previewImage() {
    const file = document.getElementById('imageUpload').files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const imgElement = document.getElementById('uploadedImage');
            imgElement.src = event.target.result;
            imgElement.style.display = "block";
        };
        reader.readAsDataURL(file);
    } else {
        alert("Please select an image first.");
    }
}

//function to simulate food classification and calorie estimation
async function generateResults() {
    const file = document.getElementById("imageUpload").files[0];
    if (!file) {
        alert("Please select an image first.");
        return;
    }

    let formData = new FormData();
    formData.append("image", file);

    try{
        let response = await fetch("http://127.0.0.1:5000/predict", { // Replace with your actual URL
            method: "POST",
            body: formData
        });
        
        //Handle restrictions here
        if (response.status === 409) {
            const result = await response.json();
            alert(result.message);
            return;
        }

        if(!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        let data = await response.json();
        console.log("API Response:", data);  // Debugging log

        if (data.processed_image_url) {
            document.getElementById('resultImage').src = data.processed_image_url;
            // Make the image visible after it's loaded
            document.getElementById('resultImage').style.display = "block";
        }else{
            throw new Error("Invalid response data.");
        }
        
          // Ensure `total_calories` exists before updating UI
          if (data.total_calories !== undefined) {
            document.getElementById('total_calories').innerText = `${data.total_calories} kcal`;
        }

        if (data.detected_items && data.detected_items.length > 0) {
            getNutritionData(data.detected_items);
        }

    // Store results only if user is logged in
           if (isUserLoggedIn === true || isUserLoggedIn === "true") {
            const originalImage = document.getElementById('uploadedImage').src;
            const processedImage = document.getElementById('resultImage').src;

            await fetch('/store_result', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    items: data.detected_items,
                    total_calories: data.total_calories,
                    image_path: originalImage,
                    result_image_path: processedImage
                })
            })
            .then(res => res.json())
            .then(data => console.log(data.message))
            .catch(err => console.error("Error storing result:", err));
        }

    }catch(error){
        console.error("Error:", error);
        //alert("Failed to get prediction. Please try again.");
        }
}

//Fetches nutrition data and triggers chart
    async function getNutritionData(foodItems) {
        let totalCarbs = 0;
        let totalProteins = 0;
        let totalFats = 0;
    
        for (const foodName of foodItems) {
            try {
                const res = await fetch(`http://127.0.0.1:5000/get_nutrition?food_name=${foodName}`);
                const nutrition = await res.json();
    
                if (nutrition && nutrition.carbs !== undefined) {
                    totalCarbs += nutrition.carbs;
                    totalProteins += nutrition.proteins;
                    totalFats += nutrition.fats;
                } else {
                    console.log("Nutrition not found for:", foodName);
                }
            } catch (error) {
                console.error("Error fetching nutrition for", foodName, error);
            }
        }
    
        // Now generate the pie chart with summed macros
        generatePieChart(totalCarbs, totalProteins, totalFats);
    }
    


//Draws Pie Chart with macros
function generatePieChart(carbs, proteins, fats) {
    const ctx = document.getElementById('nutritionChart').getContext('2d');

    // Destroy old chart if exists
    if (window.nutritionChartInstance) {
        window.nutritionChartInstance.destroy();
    }

    window.nutritionChartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Carbs', 'Proteins', 'Fats'],
            datasets: [{
                data: [carbs, proteins, fats],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',   // Carbs
                    'rgba(54, 162, 235, 0.7)',   // Proteins
                    'rgba(255, 206, 86, 0.7)'    // Fats
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Macronutrient Breakdown'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.label || '';
                            let value = context.parsed || 0;
                            return `${label}: ${value}g`;
                        }
                    }
                },
            }
        },
        plugins: [ChartDataLabels]
    });
    }
      
    //Display monthly calories of specific students
    function loadMonthlyCalorieDetails() {
        fetch('/get_monthly_details')
        .then(response => response.json())
        .then(data => {
            document.getElementById("monthly-summary-title").textContent = `Calorie Summary for ${data.month} `;  //${data.year}
            document.getElementById("monthly-total-calories").textContent = `${data.monthly_total_calories} kcal`;
    
            const tableBody = document.querySelector('#monthlyTable tbody');
            tableBody.innerHTML = ''; // Clear previous data
    
            data.records.forEach(record => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${record.date}</td>
                    <td>${record.food_name}</td>
                    <td>${record.calories}</td>
                `;
                tableBody.appendChild(row);
            });
            
            document.getElementById("calorie-summary").style.display = "block";
        }) 
        .catch(error => {
            console.error("Error loading monthly details:", error);
            document.getElementById("total-calories").textContent = "Error loading data.";
        });
    }

//Admin logs display
async function fetchLogs() {
    try {
        const response = await fetch('/admin/logs'); // This is working now
        const data = await response.json();

        const logs = data.logs;
        const tableBody = document.getElementById('logsTableBody');
        tableBody.innerHTML = ""; // Clear any previous logs

        logs.forEach(log => {
            const row = `<tr>
                            <td>${log.user_id}</td>
                            <td>${log.food_name}</td>
                            <td>${log.calories}</td>
                            <td>${log.timestamp}</td>
                        </tr>`;
            tableBody.innerHTML += row;
        });

        // Show the card
        document.getElementById('logsTableCard').style.display = 'block';
    } catch (error) {
        console.error('Error fetching logs:', error);
    }
}

document.getElementById('showLogsBtn').addEventListener('click', fetchLogs);

    //Fetch the attendance of students
    async function fetchAttendanceData() {
        const filter = document.getElementById('dateFilter').value;
    
        try {
            const response = await fetch(`/admin/attendance?filter=${filter}`);
            const data = await response.json();
    
            document.getElementById('presentCount').textContent = data.present;
            document.getElementById('absentCount').textContent = data.absent;
        } catch (error) {
            console.error('Error fetching attendance:', error);
        }
    }
    
    // Call once on page load
    document.addEventListener('DOMContentLoaded', fetchAttendanceData);
    
    //Graph display logic
    async function loadStudentAccess() {
        const month = document.getElementById("monthSelect").value;
        const response = await fetch(`/admin/student-access?month=${month}`);
        const data = await response.json();

        const list = document.getElementById("studentList");
        list.innerHTML = "";

        const studentNames = [];
        const accessCounts = [];

        data.forEach(student => {
            const li = document.createElement("li");
            li.textContent = student.name + " (ID: " + student.id + ")";
            li.style.cursor = "pointer";
            li.onclick = () => loadStudentChart(student.id, student.name, month);
            list.appendChild(li);

            studentNames.push(student.name);
            accessCounts.push(student.count);
        });
        //call for graph display
        drawAccessGraph(studentNames, accessCounts);
    }

    //Function for graph display of specific students
    function drawAccessGraph(names, counts) {
        const ctx = document.getElementById("accessChart").getContext("2d");
        if (window.accessChartInstance) window.accessChartInstance.destroy();

        // Different colors for each bar
        const backgroundColors = names.map(() => 
            `rgba(${Math.floor(Math.random()*255)}, ${Math.floor(Math.random()*255)}, ${Math.floor(Math.random()*255)}, 0.6)`
        );

        window.accessChartInstance = new Chart(ctx, {
            type: "polarArea",
            data: {
                labels: names,
                datasets: [{
                    label: "Number of Present Days",
                    data: counts,
                    backgroundColor: backgroundColors,
                    borderColor: "rgba(0, 0, 0, 0.5)",
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        //precision: 0
                    }
                },
                plugins: {
                    tooltip: {
                        enabled: true
                    },
                    legend: {
                        //display: false
                        position: 'top'
                    }
                }
            }
        });
    }

    //Loading specific student data for graph
    async function loadStudentChart(studentId, studentName, month) {
        const res = await fetch(`/admin/student/${studentId}/calories?month=${month}`);
        const data = await res.json();

        document.getElementById("studentChartCard").style.display = "block";
        document.getElementById("studentNameHeader").textContent = `Calorie Intake for ${studentName}`;

        const ctx = document.getElementById("studentChart").getContext("2d");
        if (window.studentChartInstance) window.studentChartInstance.destroy();

        // Generate a different color for each bar
        const backgroundColors = data.dates.map(() => 
            `rgba(${Math.floor(Math.random()*255)}, ${Math.floor(Math.random()*255)}, ${Math.floor(Math.random()*255)}, 0.6)`
        );

        window.studentChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: data.dates,
                datasets: [{
                    label: "Calories",
                    data: data.calories,
                    backgroundColor: backgroundColors,
                    borderColor: "rgba(0, 0, 0, 0.5)",
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    tooltip: {
                        enabled: true
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    document.addEventListener("DOMContentLoaded", loadStudentAccess);

