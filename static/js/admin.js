let cpuChart, memoryChart;

function initCharts() {
    const cpuCtx = document.getElementById('cpuChart').getContext('2d');
    const memoryCtx = document.getElementById('memoryChart').getContext('2d');
    
    cpuChart = new Chart(cpuCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'CPU Usage %',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        }
    });
    
    memoryChart = new Chart(memoryCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Memory Usage %',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        }
    });
}

function updateCharts(data) {
    // Update CPU chart
    cpuChart.data.labels = data.history.cpu.map(item => 
        new Date(item.time).toLocaleTimeString()
    );
    cpuChart.data.datasets[0].data = data.history.cpu.map(item => item.value);
    cpuChart.update();
    
    // Update Memory chart
    memoryChart.data.labels = data.history.memory.map(item => 
        new Date(item.time).toLocaleTimeString()
    );
    memoryChart.data.datasets[0].data = data.history.memory.map(item => item.value);
    memoryChart.update();
}

function updateProcessingTable(data) {
    const tbody = document.querySelector('#processTable tbody');
    tbody.innerHTML = '';
    
    data.processing.history.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(item.time).toLocaleString()}</td>
            <td>${item.files}</td>
            <td>${item.duration}s</td>
            <td>${item.status}</td>
        `;
        tbody.appendChild(row);
    });
}

document.getElementById('loadTestForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    try {
        const response = await fetch('/admin/test', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        document.getElementById('testResults').innerHTML = `
            <h3>Test Results</h3>
            <p>Average Processing Time: ${data.avgTime}s</p>
            <p>Success Rate: ${data.successRate}%</p>
            <p>Memory Usage: ${data.memoryUsage}MB</p>
        `;
    } catch (error) {
        console.error('Test failed:', error);
    }
});

// Update stats every 5 seconds
async function updateStats() {
    try {
        const response = await fetch('/admin/stats');
        const data = await response.json();
        updateCharts(data);
        updateProcessingTable(data);
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    updateStats();
    setInterval(updateStats, 5000);
});