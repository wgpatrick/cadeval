let charts = {
    successRate: null,
    checksPassed: null,
    similarity: null
};

// --- Helper Functions ---
// Simplified boolean formatting for table cells
function formatBooleanText(value) {
    if (value === true) {
        return 'Yes';
    } else if (value === false) {
        return 'No';
    } else {
        return 'N/A';
    }
}

// Simplified similarity formatting for table cells
function formatSimilarityText(value) {
    if (value === null || typeof value === 'undefined') {
        return 'N/A';
    }
    const numValue = Number(value);
    if (isNaN(numValue)) {
        return 'Invalid';
    }
    return numValue.toFixed(2) + ' mm';
}

// --- Cell Renderers for AG-Grid ---
function booleanStatusCellRenderer(params) {
    const status = formatBooleanText(params.value);
    return `<div class="cell-status-container"><span class="status-indicator status-${status.toLowerCase()}"></span><span>${status}</span></div>`;
}

function individualCheckCellRenderer(params) {
    const status = formatBooleanText(params.value);
    return `<div class="cell-status-container"><span class="status-indicator status-${status.toLowerCase()}"></span></div>`; // Only show icon for brevity
}

function similarityCellRenderer(params) {
    const formattedValue = formatSimilarityText(params.value);
    return `<span>${formattedValue}</span>`;
}


// --- Chart Creation ---
function renderSummaryCharts(metaStatistics) {
    const modelNames = Object.keys(metaStatistics);

    // Destroy existing charts if they exist
    Object.values(charts).forEach(chart => chart?.destroy());

    // Extract data for charts
    const successRates = modelNames.map(m => metaStatistics[m].overall_pipeline_success_rate); // Use the correct key
    const checksPassedRates = modelNames.map(m => metaStatistics[m].all_geo_checks_passed_rate_rel); // Use the correct key
    const avgSimilarity = modelNames.map(m => metaStatistics[m].average_similarity_distance); // Use the correct key

    const commonChartOptions = {
        scales: {
            y: { beginAtZero: true }
        },
        maintainAspectRatio: false, // Allow chart to resize within wrapper
        plugins: {
             legend: { display: false }, // Hide legend if only one dataset
             tooltip: {
                 callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) {
                             label += ': ';
                        }
                        if (context.parsed.y !== null) {
                            // Add % for rate charts
                            const suffix = context.dataset.label.includes('Rate') ? '%' : '';
                            label += context.parsed.y.toFixed(1) + suffix;
                        }
                        return label;
                    }
                 }
             }
        }
    };

    const rateChartOptions = {
         ...commonChartOptions,
        scales: { y: { beginAtZero: true, max: 100, ticks: { callback: value => value + '%' } } } // Y-axis as percentage
    };

     const similarityChartOptions = {
         ...commonChartOptions,
        scales: { y: { beginAtZero: true } }
     };


    // 1. Overall Success Rate Chart
    const successCtx = document.getElementById('successRateChart').getContext('2d');
    charts.successRate = new Chart(successCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Overall Success Rate',
                data: successRates,
                backgroundColor: 'rgba(40, 167, 69, 0.6)', // Success green
                borderColor: 'rgba(40, 167, 69, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Overall Success Rate (% Tasks Fully Passed)' }}} // Add title
    });

    // 2. Checks Passed Rate Chart
    const checksCtx = document.getElementById('checksPassedChart').getContext('2d');
    charts.checksPassed = new Chart(checksCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Checks Passed Rate (Rel. to Checks Run)', // Updated label
                // Replace nulls with NaN for Chart.js to skip them gracefully
                 data: checksPassedRates.map(d => d === null ? NaN : d),
                backgroundColor: 'rgba(255, 193, 7, 0.6)', // Warning yellow
                borderColor: 'rgba(255, 193, 7, 1)',
                borderWidth: 1
            }]
        },
        options: { ...rateChartOptions, plugins: { ...rateChartOptions.plugins, title: { display: true, text: 'Checks Passed Rate (% Rel. to Checks Run)' }}} // Updated title
    });

    // 3. Average Similarity Chart
    const similarityCtx = document.getElementById('avgSimilarityChart').getContext('2d');
     charts.similarity = new Chart(similarityCtx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Avg Similarity Distance (mm)',
                // Replace nulls with NaN for Chart.js to skip them gracefully
                data: avgSimilarity.map(d => d === null ? NaN : d),
                 backgroundColor: 'rgba(153, 102, 255, 0.6)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
         options: { ...similarityChartOptions, plugins: { ...similarityChartOptions.plugins, title: { display: true, text: 'Avg Similarity Distance (Lower is Better)' }}} // Add title
    });

    const chartsContainer = document.getElementById('charts-container');
    if (chartsContainer) {
        chartsContainer.style.display = 'flex'; // Show charts container
    } else {
        console.error("Element with ID 'charts-container' not found when trying to display charts.");
    }
}


// --- HTML Table Creation ---
const SIMILARITY_THRESHOLD_MM = 1.0; // Define pass/fail threshold for similarity

function createModelHtmlTable(modelName, modelResults, container) {
    const modelHeader = document.createElement('h2');
    modelHeader.textContent = `Model: ${modelName}`;
    container.appendChild(modelHeader);

    const table = document.createElement('table');
    table.className = 'results-table'; // Add class for styling
    const thead = table.createTHead();
    const tbody = table.createTBody();

    // Define columns - adapt these as needed from your data structure
    const columns = [
        { key: 'task_id', header: 'Task ID' },
        { key: 'scad_generation_success', header: 'SCAD Gen', format: formatBooleanText, isBoolean: true },
        { key: 'render_success', header: 'Render', format: formatBooleanText, isBoolean: true },
        // { key: 'render_status_detail', header: 'Render Status' }, // Example: Can be added back if needed
        { key: 'geometry_check_orchestration_success', header: 'Checks Run', format: formatBooleanText, isBoolean: true },
        { key: 'individual_geometry_checks_passed', header: 'Checks Passed', format: formatBooleanText, isBoolean: true },
        // Add individual check columns dynamically if they exist
        ...(modelResults[0]?.individual_check_statuses ?
            Object.keys(modelResults[0].individual_check_statuses).map(checkKey => ({
                key: `individual_check_statuses.${checkKey}`,
                header: checkKey.replace('check_', '').replace(/_/g, ' ').replace('accurate', 'Acc.').replace('successful', 'OK').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '), // Nicer Header
                format: formatBooleanText, // Use boolean formatter for display
                isBoolean: true // Mark as boolean for styling
            })) : []),
        { key: 'geometric_similarity_distance', header: 'Similarity (mm)', format: formatSimilarityText, isSimilarity: true }, // Mark for special styling
        // { key: 'geometry_check_error_detail', header: 'Check Error' }, // Example: Can be added back if needed
        // { key: 'generation_error', header: 'Generation Error' } // Example: Can be added back if needed
    ];

    // Create header row
    const headerRow = thead.insertRow();
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.header;
        headerRow.appendChild(th);
    });

    // Create data rows
    modelResults.forEach(result => {
        const row = tbody.insertRow();
        let isRowFullySuccessful = true; // Assume success initially for the row
        let taskIdCell = null; // To store the Task ID cell for later styling

        columns.forEach(col => {
            const cell = row.insertCell();
            // Handle nested keys like 'individual_check_statuses.check_manifold'
            let rawValue = col.key.split('.').reduce((obj, key) => obj?.[key], result);
            let displayValue;
            let cellStatusClass = ''; // e.g., 'status-yes', 'status-no', 'status-na'

            // Format for display
            if (col.format) {
                displayValue = col.format(rawValue);
            } else {
                displayValue = rawValue !== null && typeof rawValue !== 'undefined' ? rawValue : 'N/A';
            }
            cell.textContent = displayValue;

            // Determine cell status class for styling
            if (displayValue === 'N/A') {
                cellStatusClass = 'status-na';
                isRowFullySuccessful = false; // N/A counts as overall failure for the row
            } else if (col.isBoolean) {
                if (rawValue === true) {
                     cellStatusClass = 'status-yes';
                 } else {
                     cellStatusClass = 'status-no';
                     isRowFullySuccessful = false; // Any 'No' makes the row fail
                 }
            } else if (col.isSimilarity) {
                const numericValue = Number(rawValue);
                if (!isNaN(numericValue)) {
                    if (numericValue <= SIMILARITY_THRESHOLD_MM) {
                         cellStatusClass = 'status-yes';
                     } else {
                         cellStatusClass = 'status-no';
                         isRowFullySuccessful = false; // High similarity makes the row fail
                     }
                } else {
                    cellStatusClass = 'status-na'; // Treat non-numeric similarity as N/A
                    isRowFullySuccessful = false;
                }
            }

            // Apply the status class to the current cell
            if (cellStatusClass) {
                cell.classList.add(cellStatusClass);
            }

            // Store the Task ID cell when we encounter it
            if (col.key === 'task_id') {
                taskIdCell = cell;
            }
        });

        // After processing all cells in the row, style the Task ID cell based on overall row success
        if (taskIdCell) {
            taskIdCell.classList.add(isRowFullySuccessful ? 'status-yes' : 'status-no');
        }
    });

    container.appendChild(table);
}


// --- Dashboard Initialization (fetches data automatically) ---
async function initializeDashboard() {
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorIndicator = document.getElementById('error-indicator');
    const chartsContainer = document.getElementById('charts-container');
    const gridsContainer = document.getElementById('grids-container');

    // Show loading indicator initially, clear errors/content
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (errorIndicator) errorIndicator.style.display = 'none';
    if (chartsContainer) chartsContainer.style.display = 'none';
    if (gridsContainer) gridsContainer.innerHTML = '';

    try {
        // Fetch data (Restored)
        console.log("Fetching dashboard_data.json...");
        const response = await fetch('./dashboard_data.json');
        console.log("Fetch response status:", response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} - Could not fetch dashboard_data.json. Make sure it exists in the dashboard/ directory and you've run scripts/process_results.py.`);
        }
        const data = await response.json();
        console.log("Data loaded successfully:", data);

        // Validate data structure (basic checks)
        if (!data || typeof data !== 'object') {
             throw new Error("Invalid data format: Input is not an object.");
         }
         if (!data.results_by_model || typeof data.results_by_model !== 'object') {
             throw new Error("Invalid data format: Missing or invalid 'results_by_model'.");
         }
          if (!data.meta_statistics || typeof data.meta_statistics !== 'object') {
             throw new Error("Invalid data format: Missing or invalid 'meta_statistics'.");
         }


        // Populate header
        const runIdElement = document.getElementById('run-id');
        if (runIdElement) {
            // Use run_id from fetched data
             runIdElement.textContent = `CadEval Dashboard - Run: ${data.run_id || 'Unknown'}`;
         } else {
             console.error("Element with ID 'run-id' not found.");
         }

        // Render charts
        if (data.meta_statistics && Object.keys(data.meta_statistics).length > 0) {
            console.log("Rendering summary charts...");
            renderSummaryCharts(data.meta_statistics);
            // Ensure container is shown after rendering
            if (chartsContainer) chartsContainer.style.display = 'flex';
        } else {
            console.warn("No meta-statistics found in data.");
            if (chartsContainer) {
                 chartsContainer.innerHTML = '<p style="text-align:center; width:100%;">No summary statistics available.</p>';
                 chartsContainer.style.display = 'block'; // Show the message
            } else {
                 console.error("Element with ID 'charts-container' not found.");
             }
        }

        // Create tables
        if (gridsContainer && data.results_by_model && Object.keys(data.results_by_model).length > 0) {
            console.log("Creating results tables...");
             for (const [modelName, modelResults] of Object.entries(data.results_by_model)) {
                 // Basic check if modelResults is an array before processing
                 if (Array.isArray(modelResults)) {
                     createModelHtmlTable(modelName, modelResults, gridsContainer);
                 } else {
                      console.warn(`Skipping table for model ${modelName}: results are not an array.`);
                 }
             }
         } else if (!gridsContainer) {
             console.error("Element with ID 'grids-container' not found.");
        } else {
             console.warn("No results_by_model found in data or container missing.");
             if (gridsContainer) gridsContainer.innerHTML = '<p style="text-align:center; width:100%;">No detailed results available.</p>'; // Show message
        }

        // Hide loading indicator on success
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        console.log("Dashboard initialized successfully.");

    } catch (error) {
        console.error('Error initializing dashboard:', error);
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        if (errorIndicator) {
            errorIndicator.textContent = `Error initializing dashboard: ${error.message}`;
            errorIndicator.style.display = 'block';
        } else {
             console.error("Element with ID 'error-indicator' not found.");
         }
         // Keep charts/grids containers hidden on error
         if (chartsContainer) chartsContainer.style.display = 'none';
         if (gridsContainer) gridsContainer.innerHTML = ''; // Clear grids container on error too
    }
}

// --- Automatically Initialize on DOM Ready ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed. Initializing dashboard...");
    initializeDashboard(); // Call initialization automatically
});

// NOTE: Removed the file input handling logic.
// Initialization now happens automatically by fetching dashboard_data.json. 